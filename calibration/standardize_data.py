import os
import json
from glob import glob
from copy import deepcopy


def standardize_data(
    data_path, 
    original_data,
    question_to_name,
    questionaire,
):

    prediction_fpath_lst = glob(os.path.join(data_path, '*_parsed.json'))

    situation_to_idx = {}
    sit_idx = 0
    all_predictions = []
    for prediction_fpath in prediction_fpath_lst:
        with open(prediction_fpath, 'r') as f:
            predictions = json.load(f)
        all_predictions.extend(predictions)

    organized_predictions = [{} for _ in range(1200)]
    visited_situations = set()
    for entry in all_predictions:
        cur_situation = entry['situation']

        if cur_situation in situation_to_idx:
            cur_idx = situation_to_idx[cur_situation]
        else:
            situation_to_idx[cur_situation] = sit_idx
            cur_original_entry = [
                ele for ele in original_data if ele['situation'] == cur_situation
            ][0]
            organized_predictions[sit_idx] = cur_original_entry
            cur_idx = deepcopy(sit_idx)
            sit_idx += 1

        cur_appraisal_dim = question_to_name[entry['question']]

        if 'appraisal_d_pred_list' not in organized_predictions[cur_idx]:
            organized_predictions[cur_idx]['appraisal_d_pred_list'] = [{} for _ in range(5)]

        cur_response = [int(ele.split(':')[-1]) for ele in entry['response']][:5]

        for rating_idx, rating in enumerate(cur_response):
            if (cur_appraisal_dim == 'self_control') and \
                (cur_appraisal_dim in organized_predictions[cur_idx]['appraisal_d_pred_list'][rating_idx]):

                previous_rating = organized_predictions[cur_idx]['appraisal_d_pred_list'][rating_idx]['self_control']
                organized_predictions[cur_idx]['appraisal_d_pred_list'][rating_idx]['goal_support'] = previous_rating

                organized_predictions[cur_idx]['appraisal_d_pred_list'][rating_idx][cur_appraisal_dim] = rating

            organized_predictions[cur_idx]['appraisal_d_pred_list'][rating_idx][cur_appraisal_dim] = rating

    organized_predictions = [ele for ele in organized_predictions if 'appraisal_d_pred_list' in ele]
    return organized_predictions


if __name__ == '__main__':
    original_data = json.load(open('./hf_processed/envent_test_repeated_hf_processed.json'))

    questionaire = json.load(
        open('./prompts/envent_questionnaire_reader.json')
    )
    question_to_name = {
        question['Dquestion']: question['Dname'] for question in questionaire
    }
    appraisal_d_lst = list(question_to_name.keys())

    organized_data = standardize_data(
        data_path='./cache/llama8/temp_0.25/',
        original_data=original_data,
        question_to_name=question_to_name,
        questionaire=questionaire,
    )
    with open('./cache/llama8_fomatted.json', 'w') as f:
        json.dump(organized_data, f)

    organized_data = standardize_data(
        data_path='./cache/qwen7/temp_0.75/',
        original_data=original_data,
        question_to_name=question_to_name,
        questionaire=questionaire,
    )
    with open('./cache/qwen7_fomatted.json', 'w') as f:
        json.dump(organized_data, f)
