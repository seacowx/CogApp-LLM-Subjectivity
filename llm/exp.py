import os
import json
import yaml
import asyncio
import argparse
from copy import deepcopy
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

import torch
import random
import numpy as np

from llms import vLLMInference
from utils import make_demographic_info, make_traits_info


NUM_SAMPLE = 30

random.seed(42)

# Use 0.25 for Llama3-8B
# Use 0.75 for Qwen2.5-7B

model_temperature_map = {
    'llama8': 0.25,
    'qwen7': 0.75,
    'llama70': 1.5,
    'qwen72': 1.5,
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
):

    root_path = f'./cache/{model}/temp_{temperature}'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    root_path = os.path.join(root_path, data_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    postfix = ''
    postfix += '_demo' if add_demo else ''
    postfix += '_traits' if add_traits else ''
    postfix += '_parsed' if parsed else ''

    fpath = os.path.join(root_path, f'{chunk_idx}{postfix}.json')
    with open(fpath, 'w') as f:
        json.dump(out_data, f, indent=4)


def parse_response(cur_msg_lst, response_chunk, vllm, temperature, fix_record, question_to_name, do_fix: bool=True):
    fix_id_lst = []
    cached_situation, cached_dimension, cached_msg = [], [], []
    for idx, (cur_message, cur_response) in enumerate(zip(cur_msg_lst, response_chunk)):

        cur_response = cur_response.rsplit('|>', 1)[-1].strip()
        cur_msg_content = cur_message[-1]['content']
        cur_situation = cur_msg_content.rsplit('[Situation]', 1)[-1].split('\n\n')[0].strip()
        cur_question = cur_msg_content.rsplit('[Experiencer\'s Feeling]', 1)[-1].strip()
        cur_dimension = question_to_name[cur_question]

        if 'rating:' not in cur_response.lower():
            fix_id_lst.append(idx)

        cached_situation.append(cur_situation)
        cached_dimension.append(cur_dimension)
        cached_msg.append(cur_message)

    corrupted_situation_dimension = []
    fixed_idx_lst = []
    fixed_response_lst = []
    for idx in tqdm(fix_id_lst, desc='Fixing corrupted responses'):
        cur_situation = cached_situation[idx]
        cur_dimension = cached_dimension[idx]

        if (cur_situation, cur_dimension) in corrupted_situation_dimension:
            continue

        cur_msg = cached_msg[idx]

        if not do_fix:
            TOLERANCE = 0
        else:
            TOLERANCE = 5

        cur_response = ''
        counter = 0
        while 'rating:' not in cur_response.lower() and counter < TOLERANCE:
            cur_response = vllm.vllm_generate(
                prompts=cur_msg,
                temperature=temperature,
                progress_bar=False,
            )[0]

            cur_response = cur_response.rsplit('|>', 1)[-1].strip()
            counter += 1

        if counter == TOLERANCE:
            corrupted_situation_dimension.append((cur_situation, cur_dimension))
        else:
            fixed_idx_lst.append(idx)
            fixed_response_lst.append(cur_response)

    cur_results = []
    combo_to_idx = {}
    cur_result_idx = 0
    for idx, (cur_message, cur_response) in enumerate(zip(cur_msg_lst, response_chunk)):

        if idx in fixed_idx_lst:
            cur_response = fixed_response_lst.pop(0)

        cur_response = cur_response.rsplit('|>', 1)[-1].strip()
        cur_msg_content = cur_message[-1]['content']
        cur_situation = cur_msg_content.rsplit('[Situation]', 1)[-1].split('\n\n')[0].strip()
        cur_question = cur_msg_content.rsplit('[Experiencer\'s Feeling]', 1)[-1].strip()
        cur_dimension = question_to_name[cur_question]

        if (cur_situation, cur_dimension) in corrupted_situation_dimension:
            continue

        cur_combo = (cur_situation, cur_question, cur_dimension)
        if cur_combo in combo_to_idx:
            combo_idx = combo_to_idx[cur_combo]
            cur_results[combo_idx]['response'].append(cur_response)
        else:
            cur_results.append({
                'situation': cur_situation,
                'message': cur_message,
                'question': cur_question,
                'dimension': cur_dimension,
                'response': [cur_response],
            })
            combo_to_idx[cur_combo] = cur_result_idx
            cur_result_idx += 1

    fix_record += f'----------------------------------\n'
    fix_record += f'Running at Temperature = {temperature}\n'
    fix_record += f'Total corrupted situations: {len(set([situation for situation, _ in corrupted_situation_dimension]))}\n'
    fix_record += f'Total corrupted dimensions: {len(corrupted_situation_dimension)}\n'
    fix_record += f'Total fixed dimensions: {len(fixed_idx_lst)}\n'
    fix_record += f'----------------------------------\n\n'

    return cur_results, fix_record


async def run_exp(
    temperature: float,
    msg_chunk_list: list,
    model_name: str,
    data_name: str,
    vllm: vLLMInference,
    postfix: str,
    num_of_chunks: int,
    add_generation_prompt: bool,
    add_demo: bool,
    add_traits: bool,
    question_to_name: dict,
):
    fix_record = ''
    print('\n\n----------------------------------')
    print(f'Testing with temperature: {temperature}\n')
    for chunk_idx, msg_chunk in enumerate(msg_chunk_list):
        print(f'Processing chunk {chunk_idx+1} of {num_of_chunks}')

        if not os.path.exists(
            f'./cache/{model_name}/temp_{temperature}/{data_name}_{chunk_idx}{postfix}_parsed.json'
        ):
            if model_name == 'llama70' or model_name == 'qwen72':
                semaphore = asyncio.Semaphore(50)

                response_lst = await vllm.vllm_async_generate(
                    prompts=msg_chunk,
                    semaphore=semaphore,
                    add_generation_prompt=add_generation_prompt,
                    temperature=temperature,
                )
            else:
                response_lst = vllm.vllm_generate(
                    prompts=msg_chunk,
                    add_generation_prompt=add_generation_prompt,
                    temperature=temperature,
                )

            out_data = []
            for cur_msg, response in zip(msg_chunk, response_lst):
                out_data.append({
                    'messages': cur_msg,
                    'response': response,
                })

            cache_response(
                out_data=out_data, 
                data_name=data_name,
                chunk_idx=chunk_idx, 
                temperature=temperature, 
                model=model_name, 
                parsed=False,
                add_demo=add_demo,
                add_traits=add_traits,
            )

            do_fix = True
            if model_name == 'llama70' or model_name == 'qwen72':
                do_fix = False

            parsed_results, fix_record = parse_response(
                cur_msg_lst=msg_chunk,
                response_chunk=response_lst,
                vllm=vllm,
                temperature=temperature,
                fix_record=fix_record,
                question_to_name=question_to_name, 
                do_fix=do_fix,
            )

            cache_response(
                out_data=parsed_results, 
                data_name=data_name,
                chunk_idx=chunk_idx, 
                temperature=temperature, 
                model=model_name, 
                parsed=True,
                add_demo=add_demo,
                add_traits=add_traits,
            )

    log_postfix = '_demo' if add_demo else ''
    log_postfix += '_traits' if add_traits else ''

    if model_name == 'llama70' or model_name == 'qwen72':
        vllm.kill_server()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', required=True, type=str, help='Choose one of  ["llama8", "qwen7", "llama70", "qwen72"]'
    )
    parser.add_argument(
        '--dataset', default='envent', type=str, help='Choose between "envent", "fge", and "covidet"'
    )
    parser.add_argument(
        '--add_demo', action='store_true', help='Add demographic information to the prompt',
    )
    parser.add_argument(
        '--add_traits', action='store_true', help='Add personality traits to the prompt',
    )
    return parser.parse_args()


async def main():
    
    args = parse_args()

    if args.dataset != 'envent' and (args.add_demo or args.add_traits):
        raise ValueError('Demographic information and personality traits are only available for the Envent dataset')

    model_configs = yaml.safe_load(open('./configs/model_config.yaml', 'r'))

    assert args.model in model_configs, (
            f'Model {args.model} is not supported, ' 
            'add model info to "./configs/model_config.yaml"'
    )

    vllm = vLLMInference(
        model_name=args.model,
        quantization=model_configs[args.model]['quantization'],
    )

    # load prompt
    if args.add_demo and args.add_traits:
        reader_prompt = yaml.load(
            open('./prompts/envent_sys_prompt_with_demo_traits.yaml', 'r'), 
            Loader=yaml.FullLoader,
        )
    elif args.add_demo:
        reader_prompt = yaml.load(
            open('./prompts/envent_sys_prompt_with_demo.yaml', 'r'), 
            Loader=yaml.FullLoader,
        )
    elif args.add_traits:
        reader_prompt = yaml.load(
            open('./prompts/envent_sys_prompt_with_traits.yaml', 'r'), 
            Loader=yaml.FullLoader,
        )
    else:
        reader_prompt = yaml.load(
            open('./prompts/envent_sys_prompt_reader.yaml', 'r'), 
            Loader=yaml.FullLoader,
        )

    reader_msg = [{'role': key, 'content': value} for key, value in reader_prompt.items()]

    match args.dataset:
        case 'envent':
            data = json.load(open('../data/envent_test_repeated_hf_processed.json'))
            available_dimensions = []
        case 'fge':
            data = json.load(open('../data/fge_test_repeated_hf_processed_merged.json'))
            available_dimensions = yaml.load(
                open('../data/fge_merged_dimensions.yaml', 'r'),
                Loader=yaml.FullLoader,
            )
        case 'covidet':
            data = json.load(open('../data/covidet_test_repeated_hf_processed_merged.json'))
            available_dimensions = yaml.load(
                open('../data/covidet_merged_dimensions.yaml', 'r'),
                Loader=yaml.FullLoader,
            )
        case _:
            print('Dataset not recognized, using Envent as default')
            data = json.load(open('../data/envent_test_repeated_hf_processed.json'))
            available_dimensions = []

    CACHE_EVERY = 50 if len(data) > 100 else len(data)

    # load questionaire
    # the appraisal dimensions of FGE and COVIDET are merged to ENVENT
    # use the ENVENT questionaire for all datasets
    questionaire = json.load(open('./prompts/envent_questionnaire_reader.json', 'r'))
    if available_dimensions:
        questionaire = [
            ele for ele in questionaire if ele['Dname'] in available_dimensions
        ]
    question_to_name = {
        question['Dquestion']: question['Dname'] for question in questionaire
    }

    # set total number of questions, this is used for chuncking msg_lst
    NUM_QUESTION = len(questionaire)

    msg_lst = []
    for data_idx, data_entry in enumerate(data):
        situation = data_entry['situation']

        for demo_idx in range(NUM_SAMPLE):
            
            # add demographic information and/or personality traits
            cur_demographic_info, cur_traits_info = '', ''
            if args.add_demo or args.add_traits:
                demographic_info_lst = [
                    ele['demographic_info'] for ele in data_entry['appraisal_d_list']
                ]

                selected_demographic_info = demographic_info_lst[demo_idx]
            
                if args.add_demo:
                    cur_demographic_info = make_demographic_info(selected_demographic_info)

                if args.add_traits:
                    cur_traits_info = make_traits_info(selected_demographic_info)

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

                msg_lst.append(cur_msg)

    total_num_msg = len(msg_lst)
    print(f'Total number of messages: {total_num_msg}')

    num_of_chunks = total_num_msg // (CACHE_EVERY * NUM_SAMPLE * NUM_QUESTION)

    CHUNK_SIZE = NUM_SAMPLE * CACHE_EVERY * NUM_QUESTION
    msg_chunk_list = [
        msg_lst[i:i + CHUNK_SIZE] for i in range(0, len(msg_lst), CHUNK_SIZE)
    ]

    # add generation prompt if using Qwen2.5 models
    if 'qwen' in args.model:
        add_generation_prompt = True
    else:
        add_generation_prompt = False

    postfix = ''
    postfix += '_demo' if args.add_demo else ''
    postfix += '_traits' if args.add_traits else ''

    temperature = model_temperature_map[args.model]

    await run_exp(
        temperature=temperature,
        msg_chunk_list=msg_chunk_list,
        model_name=args.model,
        data_name=args.dataset,
        vllm=vllm,
        postfix=postfix,
        num_of_chunks=num_of_chunks,
        add_generation_prompt=add_generation_prompt,
        add_demo=args.add_demo,
        add_traits=args.add_traits,
        question_to_name=question_to_name,
    )


if __name__ == '__main__':
    asyncio.run(main())
