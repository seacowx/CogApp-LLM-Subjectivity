import os
import asyncio
import backoff
from tqdm.asyncio import tqdm as atqdm
from typing import List, Dict
from abc import ABCMeta, abstractmethod

from transformers import AutoTokenizer
import transformers.utils.logging as hf_logging
hf_logging.set_verbosity_error()

import time
import signal
import subprocess

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

os.environ["VLLM_CONFIGURE_LOGGING"] = '0'


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


    async def semaphore_inference(
        self, 
        semaphore, 
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
    ):
        async with semaphore:
            return await self.inference(
                model=model,
                message=message,
                temperature=temperature, 
                max_tokens=max_tokens, 
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                do_sample=do_sample,
                return_json=return_json,
                stream=stream,
            )


    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError, 
            openai.APIError, 
            openai.APITimeoutError, 
            openai.APIConnectionError, 
            openai.InternalServerError,
            openai.NotFoundError,
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

        if json_mode:
            response = await self.async_client.chat.completions.create(
                model='vllm-model',
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
                model='vllm-model',
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

        model_dtype=torch.bfloat16
        if model_name == 'llama3-8b':
            self.model_name = (
                '/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Meta-Llama-3-8B-Instruct/' 
                'snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a'
        )
        elif model_name == 'mixtral':
            self.model_name = (
                'mistralai/Mixtral-8x7B-Instruct-v0.1'
        )
        elif model_name == 'llama8':
            self.model_name = (
                '/scratch_tmp/prj/charnu/seacow_hf_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/' 
                'snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
            )
        elif 'qwen7' in model_name:
            self.model_name = (
                '/scratch_tmp/prj/charnu/seacow_hf_cache/models--Qwen--Qwen2.5-7B-Instruct/'
                'snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75'
            )
        elif 'llama70' in model_name:
            self.model_name = (
                '/scratch_tmp/prj/inf_llmcache/hf_cache/models--unsloth--Llama-3.3-70B-Instruct-bnb-4bit/' 
                'snapshots/74be54198eaf4f3c7fba1f4e9fa63725a810c7eb'
            )
        elif 'qwen72' in model_name:
            self.model_name = (
                '/scratch_tmp/prj/inf_llmcache/hf_cache/models--unsloth--Qwen2.5-72B-Instruct-bnb-4bit/' 
                'snapshots/95cde7b0316fd420d6fb7496c41f56fb9a1711d3'
            )

        WORLD_SIZE = torch.cuda.device_count()

        extra_kwargs = {}
        if download_dir:
            extra_kwargs['download_dir'] = download_dir
        if quantization:
            extra_kwargs['quantization'] = "bitsandbytes"
            extra_kwargs['load_format'] = "bitsandbytes"

        if quantization:
            extra_kwargs['pipeline_parallel_size'] = WORLD_SIZE
        else:
            extra_kwargs['tensor_parallel_size'] = WORLD_SIZE

        if quantization and WORLD_SIZE == 1:
            self.model = LLM(
                model=model_name,
                disable_log_stats=True,
                dtype=model_dtype,
                max_model_len=8192,
                enforce_eager=True,
                **extra_kwargs,
            )
        else:
            print(f'\n\nOffline inference not supported. Initiating server for {model_name}\n\n')

            self.server, self.model = start_vllm_server(self.model_name, world_size=WORLD_SIZE)


    def vllm_generate(
        self, 
        prompts,
        add_generation_prompt: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        progress_bar: bool = True,
    ) -> list:

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            max_tokens=max_tokens,
            skip_special_tokens=True,
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

    
    async def vllm_async_generate(
        self, 
        prompts,
        semaphore: asyncio.Semaphore,
        add_generation_prompt: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        progress_bar: bool = True,
    ):

        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # prompts = tokenizer.apply_chat_template(
        #     prompts, 
        #     tokenize=False,
        #     add_generation_prompt=add_generation_prompt,
        # )

        output =[
            self.model.semaphore_inference(
                model='vllm-model',
                semaphore=semaphore,
                message=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                ) for prompt in prompts
        ]

        output = await atqdm.gather(*output)

        return output

    
    def kill_server(self):
        kill_server(self.server)



def start_vllm_server(model_path, world_size) -> tuple:    
    server_command = [        
        'vllm',        
        'serve',        
        model_path,    
        '--served-model-name', 'vllm-model',
        '--port', '8000',
        '--api-key', 'anounymous123',
        '--load-format', 'bitsandbytes',
        '--quantization', 'bitsandbytes',
        '--max-model-len', '8192',
        '--enforce-eager', 
        '--pipeline-parallel-size', str(world_size)
    ]    

    # add api key to environemnt variable 
    os.environ['VLLM_API_KEY'] = 'anounymous123'

    server = subprocess.Popen(server_command)    
    server_running = False    
    number_of_attempts = 0    
    while (not server_running) and number_of_attempts < 100:        
        time.sleep(10)        
        result = subprocess.run(            
            [
                "curl", "http://localhost:8000/v1/models",
                "--header", "Authorization: Bearer anounymous123",
            ],             
            capture_output=True,             
            text=True,
        )        

        if 'vllm-model' in result.stdout:            
            server_running = True        

        number_of_attempts += 1    

    if not server_running:        
        kill_server(server)        
        raise Exception("vllm-server failed to start")    

    openai_server = OpenAIAsyncInference(
        base_url='http://localhost:8000/v1',
        api_key='anounymous123',
    )

    return server, openai_server


def kill_server(server):    
    server.send_signal(signal.SIGINT)    
    server.wait()    
    time.sleep(10)
