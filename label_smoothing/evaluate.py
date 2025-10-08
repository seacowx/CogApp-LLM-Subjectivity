import os
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tqdm import tqdm
from utils import (
    FileIO, 
    make_demographic_info, 
    make_traits_info,
)
from label_smoother import LabelSmoother
from eval_utils import vanilla_inference, auxiliary_inference

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import chebyshev

import torch
from datasets import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error


file_io = FileIO()


def organize_and_cache_results(
        test_data, 
        pred_rating_lst,
        sampled_rating_lst,
        sampled_smoothed_rating_lst,
        pred_prob_lst,
        appraisal_name_lst,
        modal_type,
        num_sample_lst,
        corrupted_idx_lst,
        with_demo=False,
        with_traits=False,
    ):
    
    response_idx = 0
    sample_idx = 0
    for situation_idx in range(len(test_data)):

        temp_appraisal_dict = {}
        temp_appraisal_prob_dict = {}
        temp_sampled_rating_dict = [{} for _ in range(max(num_sample_lst))]
        temp_sampled_smoothed_rating_dict = [{} for _ in range(max(num_sample_lst))]
        for appraisal_name in appraisal_name_lst:

            if response_idx in corrupted_idx_lst:
                response_idx += 1
                continue

            num_sample = num_sample_lst[sample_idx]

            temp_appraisal_dict[appraisal_name] = pred_rating_lst[response_idx] + 1
            temp_appraisal_prob_dict[appraisal_name] = pred_prob_lst[response_idx]

            cur_sampled_rating = sampled_rating_lst[response_idx]
            for i in range(num_sample):
                temp_sampled_rating_dict[i][appraisal_name] = int(cur_sampled_rating[i])

            cur_sampled_smoothed_rating = sampled_smoothed_rating_lst[response_idx]
            for i in range(num_sample):
                temp_sampled_smoothed_rating_dict[i][appraisal_name] = int(
                    cur_sampled_smoothed_rating[i]
                )

            response_idx += 1
            sample_idx += 1

        test_data[situation_idx]['appraisal_d_pred_list'] = temp_sampled_rating_dict
        test_data[situation_idx]['appraisal_d_pred_smoothed_list'] = temp_sampled_smoothed_rating_dict
        test_data[situation_idx]['appraisal_d_pred_argmax'] = temp_appraisal_dict
        test_data[situation_idx]['appraisal_d_pred_prob_list'] = temp_appraisal_prob_dict

    postfix = ''
    postfix += '_demo' if with_demo else ''
    postfix += '_traits' if with_traits else ''

    file_io.save_json(
        path = f'./cache/deberta_large_{modal_type}{postfix}_formatted.json',
        data=test_data,
    )


def echo_results(
    gt_mean_lst,
    prob_mean_lst,
    gt_var_lst,
    prob_var_lst,
    smoothed_prob_mean_lst,
    prob_wasserstein_lst,
    smoothed_prob_wasserstein_lst,
    prob_additional_metrics_lst,
    # smoothed_prob_additional_metrics_lst,
):

    prob_mae = mean_absolute_error(gt_mean_lst, prob_mean_lst)
    smoothed_prob_mae = mean_absolute_error(gt_mean_lst, smoothed_prob_mean_lst)

    prob_mse = mean_squared_error(gt_mean_lst, prob_mean_lst)
    smoothed_prob_mse = mean_squared_error(gt_mean_lst, smoothed_prob_mean_lst)

    prob_rmse = root_mean_squared_error(gt_mean_lst, prob_mean_lst)
    smoothed_prob_rmse = root_mean_squared_error(gt_mean_lst, smoothed_prob_mean_lst)

    prob_var_rmse = root_mean_squared_error(gt_var_lst, prob_var_lst)
    prob_var_mae = mean_absolute_error(gt_var_lst, prob_var_lst)

    prob_wasserstein = np.mean(prob_wasserstein_lst)
    smoothed_prob_wasserstein = np.mean(smoothed_prob_wasserstein_lst)

    print(f'MAE prob: {prob_mae:.3f}')
    print(f'MAE smoothed prob: {smoothed_prob_mae:.3f}')
    print(f'MSE prob: {prob_mse:.3f}')
    print(f'MSE smoothed prob: {smoothed_prob_mse:.3f}')
    print(f'RMSE prob: {prob_rmse:.3f}')
    print(f'RMSE smoothed prob: {smoothed_prob_rmse:.3f}')
    print(f'RMSE Var prob: {prob_var_rmse:.3f}')
    print(f'MAE Var prob: {prob_var_mae:.3f}')
    print('-'*100)
    print(f'Wasserstein prob: {prob_wasserstein:.3f}')
    print(f'Wasserstein smoothed prob: {smoothed_prob_wasserstein:.3f}')
    print('-'*100)


def additional_metrics(pred_values, label_values):
    # Clark Distance
    clark = np.sqrt(np.sum(np.square(pred_values - label_values) / np.square(pred_values + label_values)) / len(pred_values))
    # Canberra Distancec
    anberra = np.sum(np.abs(pred_values - label_values) / (np.abs(pred_values) + np.abs(label_values)))
    # Cosine Distance
    cosine_dist = cosine(pred_values, label_values)
    # Intersection Distance
    intersection = 1 - np.sum(np.minimum(pred_values, label_values)) / np.sum(np.maximum(pred_values, label_values))
    # Chebyshev Distance
    chebyshev_dist = chebyshev(pred_values, label_values[:5])

    return clark, anberra, cosine_dist, intersection, chebyshev_dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modal', type=int, default=1
    )
    parser.add_argument(
        '--dataset', type=str, default='envent', help='choose from "envent", "fge", and "covidet"'
    )
    parser.add_argument(
        '--with_demo', action='store_true'
    )
    parser.add_argument(
        '--with_traits', action='store_true'
    )
    return parser.parse_args()
        

def main():

    args = parse_args()
    np.random.seed(2024)

    if args.dataset != 'envent' and (args.with_demo or args.with_traits):
        raise ValueError('Demographic and traits information are only available for the Envent dataset.')
    
    if args.modal == 1:
        modal_type = 'unimodal'
    elif args.modal == 2:
        modal_type = 'bimodal'

    postfix = ''
    postfix += '_demo' if args.with_demo else ''
    postfix += '_traits' if args.with_traits else ''

    if args.dataset == 'envent':
        test_data = file_io.load_json(
            '../hf_processed/envent_test_repeated_hf_processed.json'
        )
        available_dimensions = []
    elif args.dataset == 'fge':
        test_data = file_io.load_json(
            '../hf_processed/fge_test_repeated_hf_processed_merged.json'
        )
        available_dimensions = file_io.load_yaml(
            '../hf_processed/fge_merged_dimensions.yaml'
        )
    elif args.dataset == 'covidet':
        test_data = file_io.load_json(
            '../hf_processed/covidet_test_repeated_hf_processed_merged.json'
        )
        available_dimensions = file_io.load_yaml(
            '../hf_processed/covidet_merged_dimensions.yaml'
        )

    # load questionaire for appraisal dimensions and filter out the dimensions not in the dataset
    questionaire = file_io.load_json(
        '../llm_baseline/prompts/envent_questionnaire_reader.json'
    )
    if available_dimensions:
        questionaire = [
            ele for ele in questionaire if ele['Dname'] in available_dimensions
        ]
    appraisal_d_to_question = {
        question['Dname']: question['Dquestion'] for question in questionaire
    }
    appraisal_d_lst = list(appraisal_d_to_question.keys())

    label_smoother = LabelSmoother()

    INSTRUCTION = (
        'Rate the narrator\'s feeling according to the given situation using the Likert scale. '
        'The scale ranges from 1 to 5 where 1 means "Not at all" and 5 means "Extremely".'
    )

    situation_lst = [situation_dict['situation'] for situation_dict in test_data]
    appraisal_d_lst_lst = [situation_dict['appraisal_d_list'] for situation_dict in test_data]

    input_output_lst = {}
    for situation_idx, cur_situation in enumerate(situation_lst):

        cur_appraisal_d_lst = appraisal_d_lst_lst[situation_idx]
        for appraisal_content in cur_appraisal_d_lst:
            if args.with_demo or args.with_traits:
                demographic_info = appraisal_content['demographic_info']
                demo_info = make_demographic_info(demographic_info)
                traits_info = make_traits_info(demographic_info)
            else:
                demo_info = ''
                traits_info = ''

            for dim_name, dim_rate in appraisal_content['appraisal_d'].items():

                cur_question = appraisal_d_to_question[dim_name]
                cur_input_context = (
                    f'{INSTRUCTION}\n\n'
                    f'Situation:\n{cur_situation}\n\n'
                    f'Narrator\'s Feeling:\n{cur_question}'
                )

                if f'{situation_idx}_{dim_name}' not in input_output_lst:
                    input_output_lst[f'{situation_idx}_{dim_name}'] = {
                        'text': cur_input_context,
                        'demo_info': demo_info,
                        'traits_info': traits_info,
                        'demo_traits_info': f'{demo_info} {traits_info}',
                        'label_lst': []
                    }

                input_output_lst[f'{situation_idx}_{dim_name}']['label_lst'].append(dim_rate)

    if args.modal == 1:
        if args.with_demo and args.with_traits:
            weight_path = (
                '/scratch_tmp/prj/charnu/ft_weights/llm_cogapp_uncertainty/' 
                'deberta_label_smoothing/unimodal_demo_traits/best_model.pt'
            )
        elif args.with_demo:
            weight_path = (
                '/scratch/prj/charnu/ft_weights/llm_cogapp_uncertainty/' 
                'deberta_label_smoothing/unimodal_demo/best_model.pt'
            )
        elif args.with_traits:
            weight_path = (
                '/scratch/prj/charnu/ft_weights/llm_cogapp_uncertainty/' 
                'deberta_label_smoothing/unimodal_traits/best_model.pt'
            )
        else:
            weight_path = (
                '/scratch_tmp/prj/charnu/ft_weights/llm_cogapp_uncertainty/' 
                'deberta_label_smoothing/unimodal/best_model.pt'
            )

    elif args.modal == 2:
        if args.with_demo and args.with_traits:
            weight_path = (
                '/scratch_tmp/prj/charnu/ft_weights/llm_cogapp_uncertainty/' 
                'deberta_label_smoothing/bimodal_demo_traits/best_model.pt'
            )
        elif args.with_demo:
            weight_path = (
                '/scratch_tmp/prj/charnu/ft_weights/llm_cogapp_uncertainty/' 
                'deberta_label_smoothing/bimodal_demo/best_model.pt'
            )
        elif args.with_traits:
            weight_path = (
                '/scratch_tmp/prj/charnu/ft_weights/llm_cogapp_uncertainty/' 
                'deberta_label_smoothing/bimodal_traits/best_model.pt'
            )
        else:
            weight_path = (
                '/scratch_tmp/prj/charnu/ft_weights/llm_cogapp_uncertainty/' 
                'deberta_label_smoothing/bimodal/best_model.pt'
            )

    print(f'Loading model from {weight_path}', end='\n\n')

    if not (args.with_demo or args.with_traits):
        gt_label_lst, pred_prob_lst, pred_rating_lst = vanilla_inference(
            weight_path=weight_path,
            input_output_lst=input_output_lst,
            modal_type=modal_type,
            data_name=args.dataset,
        )
    else:
        gt_label_lst, pred_prob_lst, pred_rating_lst = auxiliary_inference(
            weight_path=weight_path,
            input_output_lst=input_output_lst,
            modal_type=modal_type,
            max_length=512,
            batch_size=256,
            with_demo=args.with_demo,
            with_traits=args.with_traits,
        )

    gt_mean_lst, prob_mean_lst, smoothed_prob_mean_lst, num_sample_lst = [], [], [], []
    gt_var_lst, prob_var_lst = [], []
    prob_wasserstein_lst, smoothed_prob_wasserstein_lst = [], []
    prob_additional_metrics_lst, smoothed_prob_additional_metrics_lst = [], []
    sampled_rating_lst = []
    sampled_smoothed_rating_lst = []
    corrupted_idx_lst = []
    for idx, (gt_lst, prob_lst, cur_rating) in enumerate(zip(gt_label_lst, pred_prob_lst, pred_rating_lst)):

        # adjust the rating to the range of 1 to 5
        cur_rating += 1

        smoothed_prob_lst = label_smoother.smooth(
            label_lst=[cur_rating-1],
            modal=1,
        )[0]

        # correct rouding errors in prob_lst and smoothed_prob_lst
        prob_lst = [round(prob, 3) for prob in prob_lst]
        smoothed_prob_lst = [round(prob, 3) for prob in smoothed_prob_lst]

        prob_lst_residual = 1 - np.sum(prob_lst)
        smoothed_prob_lst_residual = 1 - np.sum(smoothed_prob_lst)

        prob_lst = [ele + (prob_lst_residual / len(prob_lst)) for ele in prob_lst]
        smoothed_prob_lst = [ele + (smoothed_prob_lst_residual / len(smoothed_prob_lst)) for ele in smoothed_prob_lst]

        # exclude invalid ratings
        gt_lst = [ele for ele in gt_lst if ele != -100]
        if not gt_lst:
            corrupted_idx_lst.append(idx)
            continue

        NUM_SAMPLE = len(gt_lst)

        sampled_from_prob = np.random.choice(
            np.arange(1, 6),
            p=prob_lst,
            size=NUM_SAMPLE,
        )
        sampled_from_smoothed_prob = np.random.choice(
            np.arange(1, 6),
            p=smoothed_prob_lst,
            size=NUM_SAMPLE,
        )

        gt_lst_sorted = sorted(gt_lst)
        sampled_from_prob_sorted = sorted(sampled_from_prob)
        sampled_from_smoothed_prob_sorted = sorted(sampled_from_smoothed_prob)

        num_sample_lst.append(NUM_SAMPLE)
        sampled_rating_lst.append(sampled_from_prob)
        sampled_smoothed_rating_lst.append(sampled_from_smoothed_prob)

        gt_mean = np.mean(gt_lst)
        sample_from_prob_mean = np.mean(sampled_from_prob)
        sample_from_smoothed_prob_mean = np.mean(sampled_from_smoothed_prob)

        prob_wasserstein_lst.append(wasserstein_distance(gt_lst_sorted, sampled_from_prob_sorted))
        smoothed_prob_wasserstein_lst.append(
            wasserstein_distance(gt_lst_sorted, sampled_from_smoothed_prob_sorted)
        )

        gt_mean_lst.append(gt_mean)
        prob_mean_lst.append(sample_from_prob_mean)

        gt_var_lst.append(np.var(gt_lst))
        prob_var_lst.append(np.var(sampled_from_prob))

        smoothed_prob_mean_lst.append(sample_from_smoothed_prob_mean)

    echo_results(
        gt_mean_lst=gt_mean_lst,
        prob_mean_lst=prob_mean_lst,
        gt_var_lst=gt_var_lst, 
        prob_var_lst=prob_var_lst,
        smoothed_prob_mean_lst=smoothed_prob_mean_lst,
        prob_wasserstein_lst=prob_wasserstein_lst,
        smoothed_prob_wasserstein_lst=smoothed_prob_wasserstein_lst,
        prob_additional_metrics_lst=prob_additional_metrics_lst,
    )

    organize_and_cache_results(
        test_data=test_data,
        pred_rating_lst=pred_rating_lst,
        pred_prob_lst=pred_prob_lst,
        appraisal_name_lst=appraisal_d_lst,
        modal_type=modal_type,
        sampled_rating_lst=sampled_rating_lst,
        sampled_smoothed_rating_lst=sampled_smoothed_rating_lst,
        num_sample_lst=num_sample_lst,
        corrupted_idx_lst=corrupted_idx_lst,
        with_demo=args.with_demo,
        with_traits=args.with_traits,
    )


if __name__ == '__main__':
    main()
