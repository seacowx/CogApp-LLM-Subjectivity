import os
import json
import yaml
import asyncio
import argparse
from copy import deepcopy
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

import torch
import random
import numpy as np

from llms import vLLMInference
from prompts.pair_rank_schema import PairRankSchema
from utils import make_demographic_info, make_traits_info, sanity_check

from methods.avg_conf import AvgConfParser
from methods.consistency import ConsistencyParser
from methods.pair_rank import PairRankParser


NUM_SAMPLE = 30
CACHE_EVERY = 50
TOLERANCE = 5

random.seed(42)

# Use 0.25 for Llama3-8B
# Use 0.75 for Qwen2.5-7B

model_temperature = {
    'llama8': 0.25,
    'llama3-8b': 0.25,
    'qwen7': 0.75,
}

questionaire = json.load(open('./prompts/envent_questionnaire_reader.json', 'r'))
question_to_name = {
    question['Dquestion']: question['Dname'] for question in questionaire
}


def cache_response(
    out_data, 
    data_name,
    chunk_idx, 
    temperature, 
    model, 
    parsed,
    add_demo,
    add_traits,
    eval_method,
):

    if not os.path.exists(f'./cache/{model}_{eval_method}'):
        os.makedirs(f'./cache/{model}_{eval_method}')
    if not os.path.exists(f'./cache/{model}_{eval_method}/temp_{temperature}'):
        os.makedirs(f'./cache/{model}_{eval_method}/temp_{temperature}')

    postfix = ''
    postfix += '_demo' if add_demo else ''
    postfix += '_traits' if add_traits else ''
    postfix += '_parsed' if parsed else ''

    fpath = f'./cache/{data_name}/{model}_{eval_method}/temp_{temperature}/{chunk_idx}{postfix}.json'

    if not os.path.exists(f'./cache/{data_name}/{model}_{eval_method}/'):
        os.makedirs(f'./cache/{data_name}/{model}_{eval_method}/')
    if not os.path.exists(f'./cache/{data_name}/{model}_{eval_method}/temp_{temperature}/'):
        os.makedirs(f'./cache/{data_name}/{model}_{eval_method}/temp_{temperature}/')

    with open(fpath, 'w') as f:
        json.dump(out_data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', 
        required=True, 
        type=str, 
        help='Choose between "llama8" and "qwen7"',
    )
    parser.add_argument(
        '--dataset',
        default='envent',
        type=str,
        help="Choose from \"envent\", \"covidet\", and \"fge\"",
    )
    parser.add_argument(
        '--add_demo', 
        action='store_true', 
        help='Add demographic information to the prompt',
    )
    parser.add_argument(
        '--add_traits', 
        action='store_true', 
        help='Add personality traits to the prompt',
    )
    parser.add_argument(
        '--eval_method', 
        type=str, 
        default='consistency', 
        options=['consistency', 'avg-conf', 'pair-rank'],
        help='Choose between ["consistency", "avg-conf", "pair-rank"]'
    )
    return parser.parse_args()


def main():
    
    args = parse_args()
    sanity_check(args)

    vllm = vLLMInference(
        model_name=args.model,
        quantization=False,
    )

    # load parser
    # pair-rank generates in JSON format, hence schema is needed
    schema = None
    match args.eval_method:
        case 'consistency':
            parser = ConsistencyParser()
        case 'avg-conf':
            parser = AvgConfParser()
        case 'pair-rank':
            parser = PairRankParser()
            schema = PairRankSchema

    prompt_methods = yaml.safe_load(open('./prompts/event_prompt_methods.yaml', 'r'))

    match args.eval_method:
        case 'consistency':
            prompt_postfix = prompt_methods['consistency']
        case 'avg-conf':
            prompt_postfix = prompt_methods['avg_conf']
        case 'pair-rank':
            prompt_postfix = prompt_methods['pair_rank']
        case _:
            raise ValueError('Invalid eval_method, choose from ["consistency", "avg-conf", "pair-rank"]')


    # load prompt, prompt template is universal across datasets
    if args.add_demo and args.add_traits:
        reader_prompt = yaml.safe_load(
            open('./prompts/envent_sys_prompt_with_demo_traits.yaml', 'r'), 
        )
    elif args.add_demo:
        reader_prompt = yaml.safe_load(
            open('./prompts/envent_sys_prompt_with_demo.yaml', 'r'), 
        )
    elif args.add_traits:
        reader_prompt = yaml.safe_load(
            open('./prompts/envent_sys_prompt_with_traits.yaml', 'r'), 
        )
    else:
        reader_prompt = yaml.safe_load(
            open('./prompts/envent_sys_prompt_reader.yaml', 'r'), 
        )

    # add prompt postfix corresponding to the evaluation method
    reader_prompt['user'] += ' ' + prompt_postfix 
    reader_msg = [{'role': key, 'content': value} for key, value in reader_prompt.items()]

    # load data and merged dimensions
    if args.dataset == 'envent':
        data = json.load(open('../data/envent_test_repeated_hf_processed.json'))
        available_dimensions = []
    elif args.dataset == 'covidet':
        data = json.load(open('../data/covidet_test_repeated_hf_processed_merged.json'))
        available_dimensions = yaml.load(
            open('../data/covidet_merged_dimensions.yaml', 'r'),
            Loader=yaml.FullLoader,
        )
    elif args.dataset == 'fge':
        data = json.load(open('../data/fge_test_repeated_hf_processed_merged.json'))
        available_dimensions = yaml.load(
            open('../data/fge_merged_dimensions.yaml', 'r'),
            Loader=yaml.FullLoader,
        )
    else:
        raise ValueError('Invalid dataset name, choose from "envent", "covidet", and "fge"')

    # add demographic info and personality traits is only surpported for envent datasets
    if args.dataset != 'envent' and not (not args.add_demo or not args.add_traits):
        raise ValueError('Demographic info and personality traits are only supported for the envent dataset')

    # load questionaire
    questionaire = json.load(open('./prompts/envent_questionnaire_reader.json', 'r'))

    # filter out dimensions that are not available in the dataset
    if available_dimensions:
        questionaire = [
            ele for ele in questionaire if ele['Dname'] in available_dimensions
        ]

    # number of questions in the questionaire
    NUM_QUESTION = len(questionaire)

    msg_lst = []
    for data_idx, data_entry in enumerate(data):
        situation = data_entry['situation']

        # There are 5 annotators for each situation, hence 5 demographic info
        # randomly sample a demographic info from 5 available ones
        cur_demographic_info, cur_traits_info = '', ''
        if args.add_demo or args.add_traits:
            demographic_info_lst = [
                ele['demographic_info'] for ele in data_entry['appraisal_d_list']
            ]
            selected_demographic_info = random.choice(demographic_info_lst)
            cur_demographic_info = make_demographic_info(selected_demographic_info)
            cur_traits_info = make_traits_info(selected_demographic_info)

        cur_demographic_info, cur_traits_info = '', ''

        for question_content in questionaire:

            question_desc = question_content['Dquestion']
            cur_msg = deepcopy(reader_msg)
            cur_msg[0]['content'] = cur_msg[0]['content'].replace('{{context}}', situation) \
                .replace('{{statements}}', question_desc) \
                .replace('{{demographic info}}', cur_demographic_info) \
                .replace('{{personality traits}}', cur_traits_info)
            cur_msg[1]['content'] = cur_msg[1]['content'].replace('{{context}}', situation) \
                .replace('{{statements}}', question_desc) \
                .replace('{{demographic info}}', cur_demographic_info) \
                .replace('{{personality traits}}', cur_traits_info)

            msg_lst.extend([cur_msg for _ in range(NUM_SAMPLE)])

    total_num_msg = len(msg_lst)
    print(f'Total number of messages: {total_num_msg}')

    num_of_chunks = total_num_msg // (CACHE_EVERY * NUM_SAMPLE * NUM_QUESTION)

    CHUNK_SIZE = NUM_SAMPLE * CACHE_EVERY * NUM_QUESTION
    msg_chunk_list = [
        msg_lst[i:i + CHUNK_SIZE] for i in range(0, len(msg_lst), CHUNK_SIZE)
    ]

    # add generation prompt if using Qwen2.5 models
    if args.model == 'qwen7':
        add_generation_prompt = True
    else:
        add_generation_prompt = False

    postfix = ''
    postfix += '_demo' if args.add_demo else ''
    postfix += '_traits' if args.add_traits else ''

    TEMPERATURE = model_temperature[args.model]
    fix_record = ''
    print('\n\n----------------------------------')
    print(f'Testing with temperature: {TEMPERATURE}\n')
    for chunk_idx, msg_chunk in enumerate(msg_chunk_list):
        print(f'Processing chunk {chunk_idx} of {num_of_chunks}')

        if not os.path.exists(
            f'./cache/{args.dataset}/{args.model}_{args.eval_method}/temp_{TEMPERATURE}/{chunk_idx}{postfix}_parsed.json'
        ):
            response_lst = vllm.vllm_generate(
                prompts=msg_chunk,
                add_generation_prompt=add_generation_prompt,
                temperature=TEMPERATURE,
                schema=schema,
            )

            out_data = []
            for cur_msg, response in zip(msg_chunk, response_lst):
                out_data.append({
                    'messages': cur_msg,
                    'response': response,
                })

            cache_response(
                out_data=out_data, 
                data_name=args.dataset,
                chunk_idx=chunk_idx, 
                temperature=TEMPERATURE, 
                model=args.model, 
                parsed=False,
                add_demo=args.add_demo,
                add_traits=args.add_traits,
                eval_method=args.eval_method,
            )

            parsed_results, fix_record = parser.parse(
                cur_msg_lst=msg_chunk,
                response_chunk=response_lst,
                vllm=vllm,
                temperature=TEMPERATURE,
                fix_record=fix_record,
                question_to_name=question_to_name,
                tolerance=TOLERANCE,
                schema=schema,
            )

            cache_response(
                out_data=parsed_results, 
                data_name=args.dataset,
                chunk_idx=chunk_idx, 
                temperature=TEMPERATURE, 
                model=args.model, 
                parsed=True,
                add_demo=args.add_demo,
                add_traits=args.add_traits,
                eval_method=args.eval_method,
            )


if __name__ == '__main__':
    main()
