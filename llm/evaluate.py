from copy import deepcopy
import os
import yaml
import json
import argparse
from glob import glob

import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error, mean_absolute_error


def cache_eval_result(result_dict: dict, result_path_lst: list, model_name: str, data_name: str):

    cache_result_dict = deepcopy(result_dict)

    result_postfix = ''
    if 'demo' in result_path_lst[0]:
        result_postfix += '_demo'
    if 'traits' in result_path_lst[0]:
        result_postfix += '_traits'

    for key, val in cache_result_dict.items():
        literal_val = [str(ele) for ele in val]
        cache_result_dict[key] = literal_val

    with open(f"./results/{data_name}_{model_name}{result_postfix}.json", 'w') as f:
        json.dump(cache_result_dict, f, indent=4)


def evaluate(result_path_lst: list, human_label_lst: list, question_to_name: dict, data_name: str):
    
    model_name = result_path_lst[0].split('/')[2].upper()

    # organize human label data
    situation_to_appraisal_d_lst_dict = {}
    for human_label_content in human_label_lst:

        cur_situation = human_label_content['situation'].strip()
        cur_appraisal_d_lst = human_label_content['appraisal_d_list']

        if cur_situation not in situation_to_appraisal_d_lst_dict:
            situation_to_appraisal_d_lst_dict[cur_situation] = []

        situation_to_appraisal_d_lst_dict[cur_situation].extend(cur_appraisal_d_lst)

    result_dict = {
        'appraisal_d': [],
        'human_mean': [],
        'llm_mean': [],
        'human_var': [],
        'llm_var': [],
        'human_mode': [],
        'llm_mode': [],
        'wasserstein': [],
        'chebyshev': [],
    }
    rating_comparison_by_dimension = {}
    for result_path in result_path_lst:
            
        result_lst = json.load(open(result_path))

        if not isinstance(result_lst[0], dict):
            result_lst = result_lst[0]

        for result_content in result_lst:

            cur_situation = result_content['situation']
            cur_dimension = result_content['dimension']

            cur_llm_ratings = [
                ele.rsplit(':', 1)[-1].strip() for ele in result_content['response']
            ]
            # extract digit from rating string
            cur_llm_ratings = [
                [ele for ele in rating if ele.isdigit()] for rating in cur_llm_ratings
            ]
            cur_llm_ratings = [
                ele for ele in cur_llm_ratings if ele
            ]
            try: 
                cur_llm_ratings = [
                    int(ele[0]) for ele in cur_llm_ratings
                ]
            except:
                continue

            cur_appraisal_dim = question_to_name[result_content['question']]
            cur_human_ratings = situation_to_appraisal_d_lst_dict[cur_situation.strip()]

            cur_human_ratings = [ele['appraisal_d'][cur_appraisal_dim] for ele in cur_human_ratings]
            cur_human_ratings = [ele for ele in cur_human_ratings if ele != -100]

            if not cur_human_ratings or not cur_llm_ratings:
                continue

            # make sure the length of human and llm ratings are the same
            if len(cur_human_ratings) < len(cur_llm_ratings):
                cur_llm_ratings = cur_llm_ratings[:len(cur_human_ratings)]
            else:
                cur_human_ratings = cur_human_ratings[:len(cur_llm_ratings)]

            cur_human_ratings_sorted = sorted(cur_human_ratings)
            cur_llm_ratings_sorted = sorted(cur_llm_ratings)

            if cur_dimension not in rating_comparison_by_dimension:
                rating_comparison_by_dimension[cur_dimension] = []

            rating_comparison_by_dimension[cur_dimension].append(
                np.mean(cur_human_ratings) - np.mean(cur_llm_ratings)
            )

            result_dict['appraisal_d'].append(cur_dimension)
            result_dict['human_mean'].append(np.mean(cur_human_ratings))
            result_dict['llm_mean'].append(np.mean(cur_llm_ratings))
            result_dict['human_var'].append(np.var(cur_human_ratings))
            result_dict['llm_var'].append(np.var(cur_llm_ratings))
            result_dict['human_mode'].append(np.argmax(np.bincount(cur_human_ratings)))
            result_dict['llm_mode'].append(np.argmax(np.bincount(cur_llm_ratings)))
            result_dict['wasserstein'].append(
                wasserstein_distance(cur_human_ratings_sorted, cur_llm_ratings_sorted)
            )
            result_dict['chebyshev'].append(
                dist.chebyshev(cur_human_ratings_sorted, cur_llm_ratings_sorted)
            )

    # save result_dict to file
    cache_eval_result(
        result_dict=result_dict,
        model_name=model_name,
        result_path_lst=result_path_lst,
        data_name=data_name,
    )

    rating_comparison_by_dimension = {
        k: np.mean(v) for k, v in rating_comparison_by_dimension.items()
    }

    with open(f'./results/{data_name}_by_appraisal_d.json', 'w') as f:
        json.dump(rating_comparison_by_dimension, f, indent=4)

    # compute MSE and MAE based on mean and mode
    llm_mean_mse = mean_squared_error(
        result_dict['human_mean'], 
        result_dict['llm_mean'],
    )
    llm_mode_mse = mean_squared_error(
        result_dict['human_mode'], 
        result_dict['llm_mode'],
    )

    llm_mean_rmse = np.sqrt(llm_mean_mse)
    llm_mode_rmse = np.sqrt(llm_mode_mse)

    llm_mean_mae = mean_absolute_error(
        result_dict['human_mean'], 
        result_dict['llm_mean'],
    )
    llm_var_mae = mean_absolute_error(
        result_dict['human_var'],
        result_dict['llm_var'],
    )
    llm_mode_mae = mean_absolute_error(
        result_dict['human_mode'], 
        result_dict['llm_mode'],
    )

    llm_mean_rmse = np.sqrt(llm_mean_mse)
    llm_mode_rmse = np.sqrt(llm_mode_mse)

    llm_wasserstein_distance = np.mean(result_dict['wasserstein'])
    llm_chebyshev_distance = np.mean(result_dict['chebyshev'])

    print('\n\n')
    print(f'Results for {model_name}:')
    print(f'Mean MSE: {llm_mean_mse:.4f}\nMode MSE: {llm_mode_mse:.4f}')
    print(f'Mean RMSE: {llm_mean_rmse:.4f}\nMode RMSE: {llm_mode_rmse:.4f}')
    print(f'Mean MAE: {llm_mean_mae:.4f}\nMode MAE: {llm_mode_mae:.4f}')
    print(f'Var MAE: {llm_var_mae:.4f}')
    print('\n')
    print(f'Wasserstein Distance: {llm_wasserstein_distance:.4f}')
    print(f'Chebyshev Distance: {llm_chebyshev_distance:.4f}')
    print('\n\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='envent', help='choose from "envent", "fge", and "covidet"'
    )
    parser.add_argument(
        '--baseline', type=str, default = '', help='choose between "random" and "majority". Leave empty for LLM.'
    )
    parser.add_argument(
        '--custom_fpath', type=str, default='', help='custom file path to result files'
    )
    parser.add_argument(
        '--with_demo', action='store_true', help='whether to include demographic information'
    )
    parser.add_argument(
        '--with_traits', action='store_true', help='whether to include personality traits'
    )
    parser.add_argument(
        '--model_size', type=str, default='small', help='choose from "small" and "large"'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == 'envent':
        human_label_lst = json.load(open('../data/envent_test_repeated_hf_processed.json'))
        available_dimensions = []
    elif args.dataset == 'fge':
        human_label_lst = json.load(open('../data/fge_test_repeated_hf_processed_merged.json'))
        available_dimensions = yaml.load(
            open('../data/fge_merged_dimensions.yaml', 'r'),
            Loader=yaml.FullLoader,
        )
    elif args.dataset == 'covidet':
        human_label_lst = json.load(open('../data/covidet_test_repeated_hf_processed_merged.json'))
        available_dimensions = yaml.load(
            open('../data/covidet_merged_dimensions.yaml', 'r'),
            Loader=yaml.FullLoader,
        )

    # load questionaire to map question (in prompt) to dimension name
    questionaire = json.load(open('./prompts/envent_questionnaire_reader.json', 'r'))
    if available_dimensions:
        questionaire = [
            ele for ele in questionaire if ele['Dname'] in available_dimensions
        ]
    question_to_name = {
        question['Dquestion']: question['Dname'] for question in questionaire
    }


    if args.baseline:
        baseline_result_path_lst = glob(f'./cache/{args.baseline}/{args.dataset}/0_parsed.json')
        evaluate(
            result_path_lst=baseline_result_path_lst,
            human_label_lst=human_label_lst,
            question_to_name=question_to_name,
            data_name=args.dataset,
        )
    elif args.custom_fpath:
        postfix = ''
        if args.with_demo:
            postfix += '_demo'
        if args.with_traits:
            postfix += '_traits'

        result_path_lst = glob(os.path.join(args.custom_fpath, f'*{postfix}_parsed.json'))

        evaluate(
            result_path_lst=result_path_lst,
            human_label_lst=human_label_lst,
            question_to_name=question_to_name,
            data_name=args.dataset,
        )
    else:
        postfix = ''
        if args.with_demo:
            postfix += '_demo'
        if args.with_traits:
            postfix += '_traits'

        if args.model_size == 'small':
            llama_result_path_lst = glob(f'./cache/llama8/temp_0.25/{args.dataset}/*{postfix}_parsed.json')
            qwen_result_path_lst = glob(f'./cache/qwen7/temp_0.75/{args.dataset}/*{postfix}_parsed.json')
        else:
            llama_result_path_lst = glob(f'./cache/llama70/temp_1.5/{args.dataset}/*{postfix}_parsed.json')
            qwen_result_path_lst = glob(f'./cache/qwen72/temp_1.5/{args.dataset}/*{postfix}_parsed.json')

        if args.with_demo and not args.with_traits:
            llama_result_path_lst = [
                ele for ele in llama_result_path_lst if '_traits' not in ele
            ]
            qwen_result_path_lst = [
                ele for ele in qwen_result_path_lst if '_traits' not in ele
            ]
        if args.with_traits and not args.with_demo:
            llama_result_path_lst = [
                ele for ele in llama_result_path_lst if '_demo' not in ele
            ]
            qwen_result_path_lst = [
                ele for ele in qwen_result_path_lst if '_demo' not in ele
            ]

        evaluate(
            result_path_lst=llama_result_path_lst,
            human_label_lst=human_label_lst,
            question_to_name=question_to_name,
            data_name=args.dataset,
        )
        evaluate(
            result_path_lst=qwen_result_path_lst,
            human_label_lst=human_label_lst,
            question_to_name=question_to_name,
            data_name=args.dataset,
        )



if __name__ == "__main__":
    main()
