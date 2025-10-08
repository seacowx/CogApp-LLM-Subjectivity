import ast
from tqdm import tqdm
from llms import vLLMInference
from methods.base_parser import BaseParser


class PairRankParser(BaseParser):

    def __init__(self) -> None:
        super().__init__()

    def parse(
        self,
        cur_msg_lst: list, 
        response_chunk: list, 
        vllm: vLLMInference, 
        temperature: float, 
        fix_record: str,
        question_to_name: dict,
        tolerance: int,
        schema = None,
    ):
        cur_results = []
        prev_combo = set()
        fix_id_lst = []
        cached_situation, cached_dimension, cached_msg = [], [], []
        for idx, (cur_message, cur_response) in enumerate(zip(cur_msg_lst, response_chunk)):

            cur_response = cur_response.rsplit('|>', 1)[-1].strip()
            cur_msg_content = cur_message[-1]['content']
            cur_situation = cur_msg_content.rsplit('[Situation]', 1)[-1].split('\n\n')[0].strip()
            cur_question = cur_msg_content.rsplit('[Experiencer\'s Feeling]', 1)[-1].strip()
            cur_dimension = question_to_name[cur_question]

            try:
                cur_response = ast.literal_eval(cur_response)
            except:
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

            cur_response = ''
            counter = 0
            while counter < tolerance and not cur_response:
                cur_response = vllm.vllm_generate(
                    prompts=cur_msg,
                    temperature=temperature,
                    progress_bar=False,
                    schema=schema,
                )[0]

                try:
                    cur_response = ast.literal_eval(cur_response)
                except:
                    fix_id_lst.append(idx)
                    counter += 1

            if counter == tolerance:
                corrupted_situation_dimension.append((cur_situation, cur_dimension))
            else:
                fixed_idx_lst.append(idx)
                fixed_response_lst.append(cur_response)

        for idx, (cur_message, cur_response) in enumerate(zip(cur_msg_lst, response_chunk)):

            cur_msg_content = cur_message[-1]['content']
            cur_situation = cur_msg_content.rsplit('[Situation]', 1)[-1].split('\n\n')[0].strip()
            cur_question = cur_msg_content.rsplit('[Experiencer\'s Feeling]', 1)[-1].strip()
            cur_dimension = question_to_name[cur_question]

            if (cur_situation, cur_dimension) in corrupted_situation_dimension:
                continue

            if idx in fixed_idx_lst:
                cur_response = fixed_response_lst.pop(0)

            cur_response = cur_response.rsplit('|>', 1)[-1].strip()
            cur_response = ast.literal_eval(cur_response)
            
            # match the keys of other probing methods
            cur_response['prob'] = cur_response['likelihoods']
            del cur_response['likelihoods']

            cur_combo = (cur_situation, cur_question, cur_dimension)
            if cur_combo == prev_combo:
                cur_results[-1]['response'].append(cur_response)
            else:
                cur_results.append({
                    'situation': cur_situation,
                    'message': cur_message,
                    'question': cur_question,
                    'dimension': cur_dimension,
                    'response': [cur_response],
                })

            prev_combo = cur_combo

        fix_record += f'----------------------------------\n'
        fix_record += f'Running at Temperature = {temperature}\n'
        fix_record += f'Total corrupted situations: {len(set([situation for situation, _ in corrupted_situation_dimension]))}\n'
        fix_record += f'Total corrupted dimensions: {len(corrupted_situation_dimension)}\n'
        fix_record += f'Total fixed dimensions: {len(fixed_idx_lst)}\n'
        fix_record += f'----------------------------------\n\n'

        return cur_results, fix_record