import os
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tqdm import tqdm
from utils import FileIO
from label_smoother import LabelSmoother

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
label_smoother = LabelSmoother()


def organize_and_cache_results(
        test_data, 
        pred_rating_lst,
        sampled_rating_lst,
        sampled_smoothed_rating_lst,
        pred_prob_lst,
        appraisal_name_lst,
        modal_type,
    ):
    
    response_idx = 0
    for situation_idx in range(len(test_data)):

        temp_appraisal_dict = {}
        temp_appraisal_prob_dict = {}
        temp_sampled_rating_dict = [{} for _ in range(5)]
        temp_sampled_smoothed_rating_dict = [{} for _ in range(5)]
        for appraisal_name in appraisal_name_lst:
            temp_appraisal_dict[appraisal_name] = pred_rating_lst[response_idx] + 1
            temp_appraisal_prob_dict[appraisal_name] = pred_prob_lst[response_idx]

            cur_sampled_rating = sampled_rating_lst[response_idx]
            for i in range(5):
                temp_sampled_rating_dict[i][appraisal_name] = int(cur_sampled_rating[i])

            cur_sampled_smoothed_rating = sampled_smoothed_rating_lst[response_idx]
            for i in range(5):
                temp_sampled_smoothed_rating_dict[i][appraisal_name] = int(
                    cur_sampled_smoothed_rating[i]
                )

            response_idx += 1

        test_data[situation_idx]['appraisal_d_pred_list'] = temp_sampled_rating_dict
        test_data[situation_idx]['appraisal_d_pred_smoothed_list'] = temp_sampled_smoothed_rating_dict
        test_data[situation_idx]['appraisal_d_pred_argmax'] = temp_appraisal_dict
        test_data[situation_idx]['appraisal_d_pred_prob_list'] = temp_appraisal_prob_dict

    file_io.save_json(
        path = f'./cache/deberta_large_{modal_type}_formatted.json',
        data=test_data,
    )


def echo_results(
    gt_mean_lst,
    prob_mean_lst,
    smoothed_prob_mean_lst,
    prob_wasserstein_lst,
    smoothed_prob_wasserstein_lst,
    prob_additional_metrics_lst,
    smoothed_prob_additional_metrics_lst,
):

    prob_mae = mean_absolute_error(gt_mean_lst, prob_mean_lst)
    smoothed_prob_mae = mean_absolute_error(gt_mean_lst, smoothed_prob_mean_lst)

    prob_mse = mean_squared_error(gt_mean_lst, prob_mean_lst)
    smoothed_prob_mse = mean_squared_error(gt_mean_lst, smoothed_prob_mean_lst)

    prob_rmse = root_mean_squared_error(gt_mean_lst, prob_mean_lst)
    smoothed_prob_rmse = root_mean_squared_error(gt_mean_lst, smoothed_prob_mean_lst)

    prob_wasserstein = np.mean(prob_wasserstein_lst)
    smoothed_prob_wasserstein = np.mean(smoothed_prob_wasserstein_lst)

    print(f'MAE prob: {prob_mae:.3f}')
    print(f'MAE smoothed prob: {smoothed_prob_mae:.3f}')
    print(f'MSE prob: {prob_mse:.3f}')
    print(f'MSE smoothed prob: {smoothed_prob_mse:.3f}')
    print(f'RMSE prob: {prob_rmse:.3f}')
    print(f'RMSE smoothed prob: {smoothed_prob_rmse:.3f}')
    print('-'*100)
    print(f'Wasserstein prob: {prob_wasserstein:.3f}')
    print(f'Wasserstein smoothed prob: {smoothed_prob_wasserstein:.3f}')
    print('-'*100)
    print(f'Clark prob: {np.mean(prob_additional_metrics_lst[0]):.3f}')
    print(f'Clark smoothed prob: {np.mean(smoothed_prob_additional_metrics_lst[0]):.3f}')
    print(f'Canberra prob: {np.mean(prob_additional_metrics_lst[1]):.3f}')
    print(f'Canberra smoothed prob: {np.mean(smoothed_prob_additional_metrics_lst[1]):.3f}')
    print(f'Cosine prob: {np.mean(prob_additional_metrics_lst[2]):.3f}')
    print(f'Cosine smoothed prob: {np.mean(smoothed_prob_additional_metrics_lst[2]):.3f}')
    print(f'Intersection prob: {np.mean(prob_additional_metrics_lst[3]):.3f}')
    print(f'Intersection smoothed prob: {np.mean(smoothed_prob_additional_metrics_lst[3]):.3f}')
    print(f'Chebyshev prob: {np.mean(prob_additional_metrics_lst[4]):.3f}')
    print(f'Chebyshev smoothed prob: {np.mean(smoothed_prob_additional_metrics_lst[4]):.3f}')



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
    chebyshev_dist = chebyshev(pred_values, label_values)

    return clark, anberra, cosine_dist, intersection, chebyshev_dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path', type=str, default=''
    )
    parser.add_argument(
        '--modal', type=int, default=1
    )
    return parser.parse_args()
        

def main():

    args = parse_args()
    np.random.seed(2024)
    
    if args.modal == 1:
        modal_type = 'unimodal'
    elif args.modal == 2:
        modal_type = 'bimodal'

    if args.result_path:
        prediction_file = file_io.load_json(args.result_path)
    else:
        prediction_file = f'./cache/deberta_large_{modal_type}_formatted.json'
        prediction_file = file_io.load_json(prediction_file)

    gt_label_lst, pred_prob_lst, pred_rating_lst = [], [], []
    for entry in prediction_file:

        temp_gt_label_lst = [[] for _ in range(21)]
        human_appraisal_lst = entry['appraisal_d_list']
        human_appraisal_lst = [ele['appraisal_d'] for ele in human_appraisal_lst]
        human_appraisal_lst = [list(ele.values()) for ele in human_appraisal_lst]

        for i in range(5):
            for j in range(21):
                temp_gt_label_lst[j].append(human_appraisal_lst[i][j])

        if 'appraisal_d_pred_prob_list' in entry and 'appraisal_d_pred_argmax' in entry:
            temp_prob_lst = list(entry['appraisal_d_pred_prob_list'].values())
            temp_rating_lst = list(entry['appraisal_d_pred_argmax'].values())

            gt_label_lst.extend(temp_gt_label_lst)
            pred_prob_lst.extend(temp_prob_lst)
            pred_rating_lst.extend(temp_rating_lst)

    gt_mean_lst, prob_mean_lst, smoothed_prob_mean_lst = [], [], []
    prob_wasserstein_lst, smoothed_prob_wasserstein_lst = [], []
    prob_additional_metrics_lst, smoothed_prob_additional_metrics_lst = [], []
    sampled_rating_lst = []
    sampled_smoothed_rating_lst = []
    for gt_lst, prob_lst, cur_rating in zip(gt_label_lst, pred_prob_lst, pred_rating_lst):

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

        sampled_from_prob = np.random.choice(
            np.arange(1, 6),
            p=prob_lst,
            size=5,
        )
        sampled_rating_lst.append(sampled_from_prob)

        sampled_from_smoothed_prob = np.random.choice(
            np.arange(1, 6),
            p=smoothed_prob_lst,
            size=5,
        )
        sampled_smoothed_rating_lst.append(sampled_from_smoothed_prob)

        prob_wasserstein_lst.append(wasserstein_distance(gt_lst, sampled_from_prob))
        smoothed_prob_wasserstein_lst.append(
            wasserstein_distance(gt_lst, sampled_from_smoothed_prob)
        )

        print(gt_lst)
        print(sampled_rating_lst)
        raise SystemExit()

        # # additional metrics required by YX
        # prob_additional_metrics_lst.append(
        #     additional_metrics(sampled_from_prob, gt_lst)
        # )
        # smoothed_prob_additional_metrics_lst.append(
        #     additional_metrics(sampled_from_smoothed_prob, gt_lst)
        # )

        gt_mean = np.mean(gt_lst)

        sample_from_prob_mean = np.mean(sampled_from_prob)
        sample_from_smoothed_prob_mean = np.mean(sampled_from_smoothed_prob)

        gt_mean_lst.append(gt_mean)
        prob_mean_lst.append(sample_from_prob_mean)

        smoothed_prob_mean_lst.append(sample_from_smoothed_prob_mean)

    echo_results(
        gt_mean_lst=gt_mean_lst,
        prob_mean_lst=prob_mean_lst,
        smoothed_prob_mean_lst=smoothed_prob_mean_lst,
        prob_wasserstein_lst=prob_wasserstein_lst,
        smoothed_prob_wasserstein_lst=smoothed_prob_wasserstein_lst,
        prob_additional_metrics_lst=prob_additional_metrics_lst,
        smoothed_prob_additional_metrics_lst=smoothed_prob_additional_metrics_lst,
    )


if __name__ == '__main__':
    main()
