import os
import yaml
import json
import argparse
import numpy as np
from statistics import mode
from abc import ABC, abstractmethod


class Baseline(ABC):
    def __init__(self, dataset, available_dimensions, dimension_to_question):
        self.dataset = dataset
        self.available_dimensions = available_dimensions
        self.dimension_to_question = dimension_to_question

        if not self.available_dimensions:
            self.available_dimensions = list(
                    self.dataset[0]['appraisal_d_mean'].keys()
            )

        self.results = []


    @abstractmethod
    def inference(self) -> list:
        pass


class RandomBaseline(Baseline):

    def __init__(self, dataset, available_dimensions, dimension_to_question):
        super().__init__(dataset, available_dimensions, dimension_to_question)


    def inference(self):
        for entry_dict in self.dataset:
            cur_situation = entry_dict['situation']
            for dimension in self.available_dimensions:
                cur_question = self.dimension_to_question[dimension]
                sampled_ratings = np.random.randint(1, 6, size=5)
                self.results.append({
                    'situation': cur_situation,
                    'message': 'Random baseline',
                    'question': cur_question,
                    'dimension': dimension,
                    'response': [
                        f'Rating: {rating}' for rating in sampled_ratings
                    ]
                })
        return self.results


class MajorityBaseline(Baseline):

    def __init__(self, dataset, available_dimensions, dimension_to_question):
        super().__init__(dataset, available_dimensions, dimension_to_question)

    
    def __get_majority(self) -> dict:
        """
        Calculates the majority (mode) rating for each dimension across all entries in the dataset.
        """
        all_rating_dict = {}
        for entry_dict in self.dataset:
            for dimension in self.available_dimensions:
                for rating_dict in entry_dict['appraisal_d_list']:
                    cur_rating_dict = rating_dict['appraisal_d']
                    for key, rating in cur_rating_dict.items():
                        if key in self.available_dimensions:
                            if key not in all_rating_dict:
                                all_rating_dict[key] = []
                            all_rating_dict[key].append(rating)

        for key, rating_lst in all_rating_dict.items():
            rating_lst = [ele for ele in rating_lst if ele != -100]
            all_rating_dict[key] = int(mode(rating_lst))

        return all_rating_dict



    def inference(self):
        majority_rating_dict = self.__get_majority()

        for entry_dict in self.dataset:
            cur_situation = entry_dict['situation']
            for dimension in self.available_dimensions:
                cur_question = self.dimension_to_question[dimension]
                cur_majority_rating_lst = [majority_rating_dict[dimension] for _ in range(5)]
                self.results.append({
                    'situation': cur_situation,
                    'message': 'Random baseline',
                    'question': cur_question,
                    'dimension': dimension,
                    'response': [
                        f'Rating: {rating}' for rating in cur_majority_rating_lst
                    ]
                })

        return self.results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, required=True, help='select from "fge", "envent", and "covidet'
    )
    parser.add_argument(
        '--baseline', type=str, default='random', help='choose between "random" and "majority"'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(2024)

    postfix = ''
    postfix += '_merged' if args.dataset != 'envent' else ''

    dataset = json.load(open(f'../hf_processed/{args.dataset}_test_repeated_hf_processed{postfix}.json'))
    questionaire = json.load(open('./prompts/envent_questionnaire_reader.json'))
    dimension_to_question = {
        ele['Dname']: ele['Dquestion'] for ele in questionaire
    }

    available_dimensions = []
    if args.dataset != 'envent':
        available_dimensions = yaml.load(
            open(f'../hf_processed/{args.dataset}_merged_dimensions.yaml'),
            Loader=yaml.FullLoader,
        )

    if args.baseline == 'random':
        baseline = RandomBaseline(
            dataset=dataset,
            available_dimensions=available_dimensions,
            dimension_to_question=dimension_to_question,
        )
    else:
        baseline = MajorityBaseline(
            dataset=dataset,
            available_dimensions=available_dimensions,
            dimension_to_question=dimension_to_question,
        )

    baseline_results = baseline.inference()

    root_path = f'./cache/{args.baseline}/'
    if not os.path.exists(os.path.join(root_path, args.dataset)):
        os.makedirs(os.path.join(root_path, args.dataset))

    with open(os.path.join(root_path, args.dataset, '0_parsed.json'), 'w') as f:
        json.dump(baseline_results, f, indent=4)


if __name__ == '__main__':
    main()
