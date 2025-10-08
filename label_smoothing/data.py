from typing import List, Dict, Tuple
from datasets import Dataset

import torch
import numpy as np
import torch.nn.functional as F

from label_smoother import LabelSmoother
from utils import FileIO, make_demographic_info, make_traits_info


class CogAppDataLoader:

    def __init__(self, tokenizer):
        self.file_io = FileIO()
        self.label_smoother = LabelSmoother()

        self.questionaire = self.file_io.load_json(
            '../llm_baseline/prompts/envent_questionnaire_reader.json'
        )
        self.question_to_name = {
            question['Dname']: question['Dquestion'] for question in self.questionaire
        }
        self.INSTRUCTION = (
            'Rate the narrator\'s feeling according to the given situation using the Likert scale. '
            'The scale ranges from 1 to 5 where 1 means "Not at all" and 5 means "Extremely".'
        )

        self.tokenizer = tokenizer

    def __read_data(self, data_path: str):
        return self.file_io.load_json(data_path)

    def __smooth_data_labels(self, label_lst: List[int] | List[List[int]], modal: int):
        return self.label_smoother.smooth(
            label_lst=label_lst,
            modal=modal
        )

    def __extract_input_label_pairs(
        self, 
        data: List[Dict],
        modal: int,
    ):

        input_context_lst = []
        input_demo_lst = []
        input_traits_lst = []
        input_demo_tarits_lst = []
        output_label_lst = []
        for situation_dict in data:
            cur_situation = situation_dict['situation']
            cur_appraisal_d_dict = situation_dict['appraisal_d']

            cur_demo_info = situation_dict['demographic_info']
            cur_traits_info = make_traits_info(cur_demo_info)
            cur_demo_info = make_demographic_info(cur_demo_info)

            # demographic infor and traits info are indenpendnt of appraisal dimensions
            input_demo_lst.extend([cur_demo_info]*len(cur_appraisal_d_dict))
            input_traits_lst.extend([cur_traits_info]*len(cur_appraisal_d_dict))
            input_demo_tarits_lst.extend(
                [cur_demo_info + ' ' + cur_traits_info]*len(cur_appraisal_d_dict)
            )

            for dim_name, dim_rating in cur_appraisal_d_dict.items():
                cur_question = self.question_to_name[dim_name]
                cur_input_context = (
                    f'{self.INSTRUCTION}\n\n'
                    f'Situation:\n{cur_situation}\n\n'
                    f'Narrator\'s Feeling:\n{cur_question}'
                )

                input_context_lst.append(cur_input_context)

                # shift the labels to 0-4
                if modal == 1:
                    output_label_lst.append(dim_rating-1)
                elif modal == 2:
                    dim_1 = dim_rating-1
                    dim_2 = dim_rating-3
                    if dim_2 < 0:
                        dim_2 = 4 + dim_2 

                    output_label_lst.append(
                        [dim_1, dim_2]
                    )

        return input_context_lst, output_label_lst, input_demo_lst, input_traits_lst, input_demo_tarits_lst

    def __tokenize_text_func(self, examples):
        return self.tokenizer(
            examples['text'], 
            truncation=True
        )

    def __tokenize_aux_func(self, examples):
        return self.tokenizer(
            examples['auxiliary_info'], 
            truncation=True
        )

    def load_data(self, data_path: str, modal: int | str):

        if isinstance(modal, str):
            modal = 1 if modal == 'unimodal' else 2

        data = self.__read_data(data_path)

        input_info = self.__extract_input_label_pairs(
            data=data,
            modal=modal,
        )
        input_context_lst, output_label_lst, input_demo_lst, input_traits_lst, input_demo_traits_lst = input_info

        # output_label_lst = F.one_hot(torch.tensor(output_label_lst), num_classes=5)
        output_onehot_lst = np.eye(5)[output_label_lst]
        output_smoothed_lst = self.__smooth_data_labels(
            label_lst=output_label_lst,
            modal=modal,
        )

        hf_dataset_lst = []
        hf_demo_lst, hf_traits_lst, hf_demo_traits_lst = [], [], []
        for input_context, output_onehot, output_smoothed, input_demo, input_traits, input_demo_traits in zip(
            input_context_lst, 
            output_onehot_lst, 
            output_smoothed_lst,
            input_demo_lst,
            input_traits_lst,
            input_demo_traits_lst,
        ):

            hf_dataset_lst.append({
                'text': input_context,
                'label_onehot': output_onehot,
                'labels': output_smoothed,
            })
            hf_demo_lst.append({
                'auxiliary_info': input_demo,
            })
            hf_traits_lst.append({
                'auxiliary_info': input_traits,
            })
            hf_demo_traits_lst.append({
                'auxiliary_info': input_demo_traits,
            })

        hf_dataset_len = len(hf_dataset_lst)
        hf_dataset = Dataset.from_list(hf_dataset_lst).with_format("torch")
        hf_demo_dataset = Dataset.from_list(hf_demo_lst).with_format("torch")
        hf_traits_dataset = Dataset.from_list(hf_traits_lst).with_format("torch")
        hf_demo_traits_dataset = Dataset.from_list(hf_demo_traits_lst).with_format("torch")

        # tokenize context
        hf_dataset = hf_dataset.map(
            self.__tokenize_text_func,
            batched=True,
        )

        # tokenize demographic info
        hf_demo_dataset = hf_demo_dataset.map(
            self.__tokenize_aux_func,
            batched=True,
        )

        # tokenize personality traits
        hf_traits_dataset = hf_traits_dataset.map(
            self.__tokenize_aux_func,
            batched=True,
        )

        # tokenize personality traits
        hf_demo_traits_dataset = hf_demo_traits_dataset.map(
            self.__tokenize_aux_func,
            batched=True,
        )

        hf_dataset = hf_dataset.remove_columns('text')
        hf_demo_dataset = hf_demo_dataset.remove_columns('auxiliary_info')
        hf_traits_dataset = hf_traits_dataset.remove_columns('auxiliary_info')
        hf_demo_traits_dataset = hf_demo_traits_dataset.remove_columns('auxiliary_info')

        return hf_dataset, hf_demo_dataset, hf_traits_dataset, hf_demo_traits_dataset, hf_dataset_len