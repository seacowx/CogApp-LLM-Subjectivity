import os
import backoff
from tqdm.asyncio import tqdm
from typing import List, Dict
from abc import ABCMeta, abstractmethod
import transformers.utils.logging as hf_logging
hf_logging.set_verbosity_error()

import torch
import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai import (
    RateLimitError,
    APIError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

os.environ["VLLM_CONFIGURE_LOGGING"] = '0'

MODEL_MAP = {
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'llama70-entity-attr': 'llama70-entity-attr',
}


class OpenAIBaseModel(metaclass=ABCMeta):

    def init_model(
        self,
        base_url: str,
        api_key: str,
    ) -> None:
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def init_async_model(
        self,
        base_url: str,
        api_key: str,
    ):
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    @abstractmethod
    def inference(
        self, 
        model: str,
        message: list,
        temperature: float = 1.0, 
        max_tokens: int = 1024, 
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = False,
        return_json: bool = False,
        stream: bool = False,
        json_mode: bool = False,
    ):
        ...


class OpenAIInference(OpenAIBaseModel):


    def __init__(
        self,
        base_url: str,
        api_key: str,
    ) -> None:
        self.init_model(
            base_url=base_url, 
            api_key=api_key, 
        )


    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError, 
            openai.APIError, 
            openai.APITimeoutError, 
            openai.APIConnectionError, 
            openai.InternalServerError
        ),
        max_tries=5,
        max_time=70,
    )
    def inference(
        self, 
        model: str,
        message: list,
        temperature: float = 1.0, 
        max_tokens: int = 1024, 
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = False,
        return_json: bool = False,
        stream: bool = False,
        json_mode: bool = False,
    ):

        model_name = MODEL_MAP[model]
        if json_mode:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                response_format={ "type": "json_object" },
            )
        else:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

        if return_json:
            return response 
        else:
            try:
                gen_text = response.choices[0].message.content
                return gen_text
            except:
                return ''


class OpenAIAsyncInference(OpenAIBaseModel):

    def __init__(
        self,
        base_url: str,
        api_key: str,
    ) -> None:
        self.init_async_model(
            base_url=base_url, 
            api_key=api_key, 
        )

    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError, 
            openai.APIError, 
            openai.APITimeoutError, 
            openai.APIConnectionError, 
            openai.InternalServerError
        ),
        max_tries=10,
        max_time=70,
    )
    async def inference(
        self, 
        model: str,
        message: list,
        temperature: float = 1.0, 
        max_tokens: int = 1024, 
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = False,
        return_json: bool = False,
        stream: bool = False,
        json_mode: bool = False,
    ) -> str | ChatCompletion | None:

        model_name = MODEL_MAP[model]
        if json_mode:
            response = await self.async_client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                response_format={ "type": "json_object" },
            )
        else:
            response = await self.async_client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

        if return_json:
            return response
        else:
            return response.choices[0].message.content


class vLLMInference():

    def __init__(
        self, 
        model_name: str,
        quantization: bool,
        download_dir: str = '',
    ) -> None:

        # NOTE: correct model_name to the correct path
        dtype=torch.bfloat16
        if model_name == 'llama3-8b':
            model_name = 'meta-llama/Llama-3.1-8B-Instruct'

        elif model_name == 'mixtral':
            model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

        elif 'llama70' in model_name :
            model_name = (
            '/scratch_tmp/prj/charnu/seacow_hf_cache/models--unsloth--Meta-Llama-3.1-70B-Instruct-bnb-4bit/' 
            'snapshots/3e0db69a642d4c235a9f28f8ebac012efa0d8113'
            )
        
        elif 'qwen7' in model_name:
            model_name = 'Qwen/Qwen2.5-7B-Instruct'

        WORLD_SIZE = torch.cuda.device_count()

        extra_kwargs = {}
        if download_dir:
            extra_kwargs['download_dir'] = download_dir
        if quantization:
            extra_kwargs['quantization'] = "bitsandbytes"
            extra_kwargs['load_format'] = "bitsandbytes"

        self.model = LLM(
            model=model_name,
            disable_log_stats=True,
            dtype=dtype,
            tensor_parallel_size= WORLD_SIZE,
            max_model_len=8192,
            **extra_kwargs,
        )

    def vllm_generate(
        self, 
        prompts,
        schema = None,
        add_generation_prompt: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        progress_bar: bool = True,
    ) -> list:

        guided_decoding_params = None
        if schema:
            guided_decoding_params = GuidedDecodingParams(json=schema.model_json_schema())

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            max_tokens=max_tokens,
            skip_special_tokens=True,
            guided_decoding=guided_decoding_params,
        )
        
        tokenizer = self.model.get_tokenizer()
        prompts = tokenizer.apply_chat_template(
            prompts, 
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        outputs = self.model.generate(
            prompts, 
            self.sampling_params,
            use_tqdm=progress_bar,
        )
        out = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # out.append({
            #     'prompt': prompt,
            #     'generated_text': generated_text,
            # })

            out.append(generated_text)

        return out