from abc import ABC, abstractmethod

from llms import vLLMInference


class BaseParser(ABC):

    @abstractmethod
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
        pass