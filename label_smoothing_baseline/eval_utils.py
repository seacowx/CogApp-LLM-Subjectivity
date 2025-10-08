import os
import torch
from tqdm import tqdm

import pandas as pd
from datasets import Dataset

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding
)

from data import CogAppDataLoader
from dual_encoder import DualEncoder
from utils import (
    FileIO, 
    create_dataloader, 
)


file_io = FileIO()


def sample_ratings(prob_lst, gt_lst) -> list:

    sampled_rating_lst = []
    for cur_prob_lst, cur_gt_lst in zip(prob_lst, gt_lst):

        # correct rouding errors in prob_lst and smoothed_prob_lst
        cur_prob_lst = [round(prob, 3) for prob in cur_prob_lst]

        prob_lst_residual = 1 - np.sum(cur_prob_lst)

        cur_prob_lst = [ele + (prob_lst_residual / len(cur_prob_lst)) for ele in cur_prob_lst]

        NUM_SAMPLE = len(cur_gt_lst)

        # exclude invalid ratings
        cur_gt_lst = [ele for ele in cur_gt_lst if ele != -100]
        if not cur_gt_lst:
            sampled_from_prob = []

        sampled_from_prob = list(np.random.choice(
            np.arange(1, 6),
            p=cur_prob_lst,
            size=NUM_SAMPLE,
        ))

        sampled_rating_lst.append(sampled_from_prob)


    return sampled_rating_lst   


@torch.no_grad()
def vanilla_inference(
    weight_path: str,
    data_name: str,
    input_output_lst: dict,
    modal_type: str,
):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
    model = AutoModelForSequenceClassification.from_pretrained(
        'microsoft/deberta-v3-large',
        num_labels=5,
    )
    model.load_state_dict(torch.load(weight_path))
    model = model.to('cuda')

    input_context_lst = [
        input_output_dict['text'] for input_output_dict in input_output_lst.values()
    ]
    input_appraisal_d_lst = [
        ele.rsplit('_')[-1] for ele in input_output_lst.keys()
    ]

    input_context_lst = tokenizer(
        input_context_lst,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to('cuda')

    input_dataset = Dataset.from_dict({
        'input_ids': input_context_lst['input_ids'],
        'attention_mask': input_context_lst['attention_mask'],
    }).with_format("torch")

    input_dataloader = DataLoader(
        input_dataset,
        batch_size=128,
        shuffle=False,
    )

    gt_label_lst = [
        input_output_dict['label_lst'] for input_output_dict in input_output_lst.values()
    ]

    pred_prob_lst = []
    pred_rating_lst = []
    sampled_rating_lst = []
    for batch in tqdm(input_dataloader):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        predicted_label_lst = model(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
        ).logits

        predicted_label_lst = F.softmax(predicted_label_lst, dim=-1)
        predicted_rating_lst = torch.argmax(predicted_label_lst, dim=-1)
        predicted_label_lst = predicted_label_lst.to('cpu').numpy().tolist()
        predicted_rating_lst = predicted_rating_lst.to('cpu').numpy().tolist()

        sampled_ratings = sample_ratings(
            prob_lst=predicted_label_lst,
            gt_lst=gt_label_lst,
        )

        pred_prob_lst.extend(predicted_label_lst)
        pred_rating_lst.extend(predicted_rating_lst)
        sampled_rating_lst.extend(sampled_ratings)

    root_path = f'./cache/deberta_{modal_type}/{data_name}'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    file_io.save_json(
        os.path.join(root_path, f'deberta_large_{modal_type}_gt_label.json'),
        gt_label_lst,
    )

    file_io.save_json(
        os.path.join(root_path, f'deberta_large_{modal_type}_pred_prob.json'),
        pred_prob_lst,
    )

    file_io.save_json(
        os.path.join(root_path, f'deberta_large_{modal_type}_pred_rating.json'),
        pred_rating_lst,
    )

    # organize results
    organized_output = []
    for cur_appraisal_d, cur_gt_label, cur_pred_rating_lst, cur_sampled_rating_lst in zip(
        input_appraisal_d_lst,
        gt_label_lst,
        pred_rating_lst,
        sampled_rating_lst,
    ):

        organized_output.append(
            {
                'appraisal_d': cur_appraisal_d,
                'gt_mean': np.mean(cur_gt_label),
                'gt_var': np.var(cur_gt_label),
                'pred_mean': np.mean(cur_sampled_rating_lst),
                'pred_var': np.var(cur_sampled_rating_lst),
                'gt_answer': cur_gt_label,
                'pred_rating_lst': cur_pred_rating_lst,
                'sampled_rating_lst': cur_sampled_rating_lst,
            }
        )

    organized_output_df = pd.DataFrame(organized_output)
    organized_output_df.to_csv(
        f"./results/deberta_large_{modal_type}_results.csv",
        index=False,
        sep='\t',
    )

    return gt_label_lst, pred_prob_lst, pred_rating_lst


@torch.no_grad()
def auxiliary_inference(
    weight_path: str,
    input_output_lst: dict,
    modal_type: str,
    max_length: int,
    batch_size: int,
    with_demo: bool,
    with_traits: bool,
):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
    model = DualEncoder()
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    input_context_lst = [
        input_output_dict['text'] for input_output_dict in input_output_lst.values()
    ]
    input_appraisal_d_lst = [
        ele.rsplit('_')[-1] for ele in input_output_lst.keys()
    ]

    if with_demo and with_traits:
        input_aux_lst = [
            input_output_dict['demo_traits_info'] for input_output_dict in input_output_lst.values()
        ]
    elif with_demo:
        input_aux_lst = [
            input_output_dict['demo_info'] for input_output_dict in input_output_lst.values()
        ]
    elif with_traits:
        input_aux_lst = [
            input_output_dict['traits_info'] for input_output_dict in input_output_lst.values()
        ]

    input_context_lst = tokenizer(
        input_context_lst,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to('cuda')

    input_aux_lst = tokenizer(
        input_aux_lst,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to('cuda')

    input_dataset = Dataset.from_dict({
        'input_ids': input_context_lst['input_ids'],
        'attention_mask': input_context_lst['attention_mask'],
    }).with_format("torch")

    input_aux_dataset = Dataset.from_dict({
        'input_ids': input_aux_lst['input_ids'],
        'attention_mask': input_aux_lst['attention_mask'],
    }).with_format("torch")

    input_dataloader = DataLoader(
        input_dataset,
        batch_size=256,
        shuffle=False,
    )
    input_aux_dataloader = DataLoader(
        input_aux_dataset,
        batch_size=256,
        shuffle=False,
    )

    gt_label_lst = [
        input_output_dict['label_lst'] for input_output_dict in input_output_lst.values()
    ]

    model = model.to('cuda')

    pred_prob_lst = []
    pred_rating_lst = []
    sampled_rating_lst = []
    for batch, aux_batch in tqdm(zip(input_dataloader, input_aux_dataloader), total=len(input_dataloader)):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        aux_batch = {k: v.to('cuda') for k, v in aux_batch.items()}

        predicted_label_lst = model(
            batch['input_ids'],
            batch['attention_mask'],
            aux_batch['input_ids'],
            aux_batch['attention_mask'],
        )

        predicted_label_lst = F.softmax(predicted_label_lst, dim=-1)

        predicted_rating_lst = torch.argmax(predicted_label_lst, dim=-1)
        predicted_label_lst = predicted_label_lst.to('cpu').numpy().tolist()
        predicted_rating_lst = predicted_rating_lst.to('cpu').numpy().tolist()

        sampled_ratings = sample_ratings(
            prob_lst=predicted_label_lst,
            gt_lst=gt_label_lst,
        )

        pred_prob_lst.extend(predicted_label_lst)
        pred_rating_lst.extend(predicted_rating_lst)
        sampled_rating_lst.extend(sampled_ratings)

    postfix = ''
    postfix += '_demo' if with_demo else ''
    postfix += '_traits' if with_traits else ''

    file_io.save_json(
        f'./cache/deberta_{modal_type}{postfix}/deberta_large_{modal_type}_gt_label.json',
        gt_label_lst,
    )

    file_io.save_json(
        f'./cache/deberta_{modal_type}{postfix}/deberta_large_{modal_type}_pred_prob.json',
        pred_prob_lst,
    )

    file_io.save_json(
        f'./cache/deberta_{modal_type}{postfix}/deberta_large_{modal_type}_pred_rating.json',
        pred_rating_lst,
    )

    # organize results
    organized_output = []
    for cur_appraisal_d, cur_gt_label, cur_pred_rating_lst, cur_sampled_rating_lst in zip(
        input_appraisal_d_lst,
        gt_label_lst,
        pred_rating_lst,
        sampled_rating_lst,
    ):

        organized_output.append(
            {
                'appraisal_d': cur_appraisal_d,
                'gt_mean': np.mean(cur_gt_label),
                'gt_var': np.var(cur_gt_label),
                'pred_mean': np.mean(cur_sampled_rating_lst),
                'pred_var': np.var(cur_sampled_rating_lst),
                'gt_answer': cur_gt_label,
                'pred_rating_lst': cur_pred_rating_lst,
                'sampled_rating_lst': cur_sampled_rating_lst,
            }
        )

    organized_output_df = pd.DataFrame(organized_output)
    organized_output_df.to_csv(
        f"./results/deberta_large_{modal_type}{postfix}_results.csv",
        index=False,
        sep='\t',
    )

    return gt_label_lst, pred_prob_lst, pred_rating_lst
