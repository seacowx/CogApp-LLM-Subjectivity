"""
Code to post-process the results of pair-rank prompting is modified from the following code snippet:
https://github.com/MiaoXiong2320/llm-uncertainty/blob/main/vis_aggregated_conf_top_k.py#L334
"""

import os
import json
from tqdm import tqdm
from glob import glob

import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

import scipy
import argparse
import numpy as np
import scipy.spatial.distance as dist

from aggregators.avg_conf import AvgConfAggregator
from aggregators.pair_rank import PairRankAggregator


# load questionaire to map question (in prompt) to dimension name
questionaire = json.load(open('./prompts/envent_questionnaire_reader.json', 'r'))
question_to_name = {
    question['Dquestion']: question['Dname'] for question in questionaire
}

# ## metrics from YX
# ## Clark Distance
# clark = np.sqrt(
#     np.sum(np.square(pred_values - label_values) / np.square(pred_values + label_values)) / len(pred_values)
# )

# # # Canberra Distance
# canberra = np.sum(
#     np.abs(pred_values - label_values) / (np.abs(pred_values) + np.abs(label_values))
# )

# # # Cosine Distance
# cosine_dist = cosine(
#     pred_values, label_values
# )

# # # Intersection Distance
# intersection = 1 - np.sum(np.minimum(pred_values, label_values)) / np.sum(np.maximum(pred_values, label_values))


def evaluate_consistency(result_path_lst: list, human_label_lst: list):

    model_name = result_path_lst[0].split('/')[2].upper()

    # organize human label data
    situation_to_appraisal_d_lst_dict = {}
    for human_label_content in human_label_lst:

        cur_situation = human_label_content['situation']
        cur_appraisal_d_lst = human_label_content['appraisal_d_list']

        if cur_situation not in situation_to_appraisal_d_lst_dict:
            situation_to_appraisal_d_lst_dict[cur_situation] = []

        situation_to_appraisal_d_lst_dict[cur_situation].extend(cur_appraisal_d_lst)

    result_dict = {
        'human_mean': [],
        'llm_mean': [],
        'human_mode': [],
        'llm_mode': [],
        'wasserstein': [],
        'chebyshev': [],
    }
    for result_path in result_path_lst:
        result_lst = json.load(open(result_path))

        for result_content in result_lst:

            cur_situation = result_content['situation']
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
            cur_llm_ratings = [
                int(ele[0]) for ele in cur_llm_ratings
            ]

            cur_appraisal_dim = question_to_name[result_content['question']]

            cur_human_ratings = situation_to_appraisal_d_lst_dict[cur_situation]
            cur_human_ratings = [ele['appraisal_d'][cur_appraisal_dim] for ele in cur_human_ratings]

            cur_human_ratings_sorted = sorted(cur_human_ratings)
            cur_llm_ratings_sorted = sorted(cur_llm_ratings)

            result_dict['human_mean'].append(np.mean(cur_human_ratings))
            result_dict['llm_mean'].append(np.mean(cur_llm_ratings))
            result_dict['human_mode'].append(np.argmax(np.bincount(cur_human_ratings)))
            result_dict['llm_mode'].append(np.argmax(np.bincount(cur_llm_ratings)))
            result_dict['wasserstein'].append(
                scipy.stats.wasserstein_distance(cur_human_ratings_sorted[:5], cur_llm_ratings_sorted)
            )
            result_dict['chebyshev'].append(
                dist.chebyshev(cur_human_ratings_sorted, cur_llm_ratings_sorted)
            )
            
            echo_results(
                result_dict=result_dict,
                model_name=model_name,
            )


def evaluate_avg_conf(result_path_lst: list, human_label_lst: list, aggregator: AvgConfAggregator | None):

    model_name = result_path_lst[0].split('/')[3].upper()

    # defaults to envent
    NUM_MEASUREMENTS = 5
    if 'covidet' in result_path_lst[0]:
        NUM_MEASUREMENTS = 2
    elif 'fge' in result_path_lst[0]:
        NUM_MEASUREMENTS = 15

    # organize human label data
    situation_to_appraisal_d_lst_dict = {}
    for human_label_content in human_label_lst:

        cur_situation = human_label_content['situation']
        cur_appraisal_d_lst = human_label_content['appraisal_d_list']

        if cur_situation not in situation_to_appraisal_d_lst_dict:
            situation_to_appraisal_d_lst_dict[cur_situation] = []

        situation_to_appraisal_d_lst_dict[cur_situation].extend(cur_appraisal_d_lst)

    result_dict = {
        'human_mean': [],
        'llm_mean': [],
        'human_var': [],
        'llm_var': [],
        'human_mode': [],
        'llm_mode': [],
        'wasserstein': [],
        'chebyshev': [],
    }
    for result_path in result_path_lst:
        result_lst = json.load(open(result_path))

        for result_content in result_lst:

            cur_situation = result_content['situation']
            cur_llm_ratings = result_content['response']

            if aggregator:
                aggregated_rating_dict = aggregator.aggregate(
                    rating_prob_dict=cur_llm_ratings,
                )

                aggregated_rating_dist = [0] * 5
                for rating, prob in aggregated_rating_dict.items():
                    # invalid rating, range is 1-5
                    if rating > 5:
                        continue
                    aggregated_rating_dist[rating - 1] = prob

                # allocate extra density uniformly if the sum of probabilities is less than 1
                if sum(aggregated_rating_dist) < 1:
                    extra_density = (1 - sum(aggregated_rating_dist)) / 5
                    aggregated_rating_dist = [
                        ele + extra_density for ele in aggregated_rating_dist
                    ]

                # sample according to the aggregated rating probabilities
                cur_llm_ratings = np.random.choice(
                    np.arange(1, 6, 1), size=NUM_MEASUREMENTS, p=aggregated_rating_dist,
                )
            else:
                cur_llm_ratings = [ele['rating'] for ele in cur_llm_ratings]

            cur_appraisal_dim = question_to_name[result_content['question']]

            cur_human_ratings = situation_to_appraisal_d_lst_dict.get(cur_situation, [])
            cur_human_ratings = [ele['appraisal_d'][cur_appraisal_dim] for ele in cur_human_ratings]
            cur_human_ratings = [ele for ele in cur_human_ratings if ele != -100]

            if not cur_human_ratings:
                continue

            # in case of duplicated context
            if len(cur_human_ratings) > NUM_MEASUREMENTS:
                cur_human_ratings = cur_human_ratings[:NUM_MEASUREMENTS]

            # truncate llm ratings to match the length of human ratings. 
            if len(cur_llm_ratings) < len(cur_human_ratings):
                raise ValueError(
                    'Insufficient sample for LLM ratings.'
                    'LLM rating samples need to be at least as many as that of human ratings\n\n'
                    f'Number of human ratings: {len(cur_human_ratings)}\nNumber of LLM ratings: {len(cur_llm_ratings)}'
                )

            cur_llm_ratings = cur_llm_ratings[:len(cur_human_ratings)]

            cur_human_ratings_sorted = sorted(cur_human_ratings)
            cur_llm_ratings_sorted = sorted(cur_llm_ratings)

            result_dict['human_mean'].append(np.mean(cur_human_ratings))
            result_dict['llm_mean'].append(np.mean(cur_llm_ratings))
            result_dict['human_var'].append(np.var(cur_human_ratings))
            result_dict['llm_var'].append(np.var(cur_llm_ratings))
            result_dict['human_mode'].append(np.argmax(np.bincount(cur_human_ratings)))
            result_dict['llm_mode'].append(np.argmax(np.bincount(cur_llm_ratings)))
            result_dict['wasserstein'].append(
                scipy.stats.wasserstein_distance(cur_human_ratings_sorted, cur_llm_ratings_sorted)
            )
            result_dict['chebyshev'].append(
                dist.chebyshev(cur_human_ratings_sorted, cur_llm_ratings_sorted)
            )
            
    echo_results(
        result_dict=result_dict,
        model_name=model_name,
    )


def evaluate_pair_rank(
        result_path_lst: list, 
        human_label_lst: list, 
        num_samples: int,
        aggregator: PairRankAggregator | None
    ):

    model_name = result_path_lst[0].split('/')[2].upper()

    # defaults to envent
    NUM_MEASUREMENTS = 5
    if 'covidet' in result_path_lst[0]:
        NUM_MEASUREMENTS = 2
    elif 'fge' in result_path_lst[0]:
        NUM_MEASUREMENTS = 15

    # organize human label data
    situation_to_appraisal_d_lst_dict = {}
    for human_label_content in human_label_lst:

        cur_situation = human_label_content['situation']
        cur_appraisal_d_lst = human_label_content['appraisal_d_list']

        if cur_situation not in situation_to_appraisal_d_lst_dict:
            situation_to_appraisal_d_lst_dict[cur_situation] = []

        situation_to_appraisal_d_lst_dict[cur_situation].extend(cur_appraisal_d_lst)

    result_dict = {
        'human_mean': [],
        'llm_mean': [],
        'human_var': [],
        'llm_var': [],
        'human_mode': [],
        'llm_mode': [],
        'wasserstein': [],
        'chebyshev': [],
    }
    for result_path in tqdm(result_path_lst, position=0, leave=False):
        result_lst = json.load(open(result_path))

        for result_content in tqdm(result_lst, position=1, leave=False):

            cur_situation = result_content['situation']
            cur_llm_ratings = result_content['response']

            if aggregator:
                # adjust number of samples to be used for computing categorical distribution
                cur_llm_ratings = cur_llm_ratings[:num_samples]
                # llm generated rank could be corrupted, especially for small LLMs
                try:
                    _, _, aggregated_rating_dist = aggregator.aggregate(
                        rating_prob_dict=cur_llm_ratings,
                    )
                except:
                    continue

                # sample according to the aggregated rating probabilities
                cur_llm_ratings = np.random.choice(
                    np.arange(1, 6, 1), size=NUM_MEASUREMENTS, p=aggregated_rating_dist,
                )
            else:
                cur_rating_mtx = np.zeros((len(cur_llm_ratings), 5))
                for idx, rating_prob_dict in enumerate(cur_llm_ratings):
                    try:
                        cur_rating_mtx[idx] = rating_prob_dict['ranked_ratings']
                    except:
                        continue

                cur_rating_mtx = cur_rating_mtx[~np.all(cur_rating_mtx == 0, axis=1)]
                # cur_llm_ratings = scipy.stats.mode(cur_rating_mtx, axis=0).mode
                # cur_llm_ratings = cur_llm_ratings.astype(int).tolist()
                cur_llm_ratings = [
                    ele['ranked_ratings'][0] for ele in cur_llm_ratings
                ]

            cur_appraisal_dim = question_to_name[result_content['question']]

            cur_human_ratings = situation_to_appraisal_d_lst_dict.get(cur_situation, [])
            cur_human_ratings = [ele['appraisal_d'][cur_appraisal_dim] for ele in cur_human_ratings]
            cur_human_ratings = [ele for ele in cur_human_ratings if ele != -100]

            if not cur_human_ratings:
                continue

            # truncate llm ratings to match the length of human ratings.
            cur_llm_ratings = cur_llm_ratings[:len(cur_human_ratings)]
            # in case of duplicated context, also truncate human ratings to match the length of llm ratings
            cur_human_ratings = cur_human_ratings[:len(cur_llm_ratings)]

            cur_human_ratings_sorted = sorted(cur_human_ratings)
            cur_llm_ratings_sorted = sorted(cur_llm_ratings)

            result_dict['human_mean'].append(np.mean(cur_human_ratings))
            result_dict['llm_mean'].append(np.mean(cur_llm_ratings))
            result_dict['human_var'].append(np.var(cur_human_ratings))
            result_dict['llm_var'].append(np.var(cur_llm_ratings))
            result_dict['human_mode'].append(np.argmax(np.bincount(cur_human_ratings)))
            result_dict['llm_mode'].append(np.argmax(np.bincount(cur_llm_ratings)))
            result_dict['wasserstein'].append(
                scipy.stats.wasserstein_distance(cur_human_ratings_sorted, cur_llm_ratings_sorted)
            )
            result_dict['chebyshev'].append(
                dist.chebyshev(cur_human_ratings_sorted, cur_llm_ratings_sorted)
            )
            
    echo_results(
        result_dict=result_dict,
        model_name=model_name,
    )


def echo_results(result_dict: dict, model_name: str):

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
    llm_mode_mae = mean_absolute_error(
        result_dict['human_mode'], 
        result_dict['llm_mode'],
    )

    llm_var_mae = mean_absolute_error(
        result_dict['human_var'],
        result_dict['llm_var'],
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
    parser = argparse.ArgumentParser(description='Evaluate the results of the verbalizer-uncertainty methods')
    parser.add_argument( 
        '--data_path', 
        type=str, 
        default='./cache/llama8_consistency/temp_0.25/'
    )
    parser.add_argument(
        '--with_demo',
        action='store_true',
        help='add demographic information',
    )
    parser.add_argument(
        '--with_traits',
        action='store_true',
        help='add personality traits',
    )
    parser.add_argument(
        '--with_aggregator', 
        action='store_true', 
        help='Whether to use aggregator for evaluation',
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=30, 
        help='Number of samples to use for computing categorical distribution in pair-rank',
    )
    return parser.parse_args()


def main():

    args = parse_args()
    eval_method = args.data_path.split('/')[3].split('_', 1)[-1].split('_')[0]

    result_path_lst = glob(os.path.join(args.data_path, '*_parsed.json'))

    if args.with_demo and args.with_traits:
        result_path_lst = [
            ele for ele in result_path_lst if 'demo_traits' in ele
        ]
    elif args.with_demo:
        result_path_lst = [
            ele for ele in result_path_lst if 'demo_' in ele
        ]
    elif args.with_traits:
        result_path_lst = [
            ele for ele in result_path_lst if 'traits_' in ele
        ]
    else:
        result_path_lst = [
            ele for ele in result_path_lst if ('demo_' not in ele and 'traits_' not in ele)
        ]

    if 'envent' in args.data_path:
        human_label_lst = json.load(open('../hf_processed/envent_test_repeated_hf_processed.json'))
    elif 'covidet' in args.data_path:
        human_label_lst = json.load(open('../hf_processed/covidet_test_repeated_hf_processed_merged.json')) 
    elif 'fge' in args.data_path:
        human_label_lst = json.load(open('../hf_processed/fge_test_repeated_hf_processed_merged.json'))
    else:
        raise ValueError('Unknown dataset, choose from envent, covidet, fge')

    aggregator = None
    if eval_method == 'consistency':
        evaluate_consistency(
            result_path_lst=result_path_lst,
            human_label_lst=human_label_lst,
        )

    elif eval_method == 'avg-conf':
        if args.with_aggregator:
            aggregator = AvgConfAggregator()
        evaluate_avg_conf(
            result_path_lst=result_path_lst,
            human_label_lst=human_label_lst,
            aggregator=aggregator,
        )
    elif eval_method == 'pair-rank':
        if args.with_aggregator:
            aggregator = PairRankAggregator()
        evaluate_pair_rank(
            result_path_lst=result_path_lst,
            human_label_lst=human_label_lst,
            aggregator=aggregator,
            num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()
