import torch
from typing import cast, Any
from .data import MinimalSource
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Model(BaseModel):
    llm_name: str = 'Qwen/Qwen3-0.6B'
    _llm: Any = None
    tokenizer: Any = None

    def prompt(self, query: str, data: list[MinimalSource]) -> list[dict]:
        content: list = []
        for i in data:
            content.append(i.get_content())
        messages = [
            {'role': 'user', 'content': f'Hello, you are a rag assistant who '
             f'can help me answer in one sentence the query({query}) '
             f'from the retrieved data sources in {content}: '}
        ]
        return messages

    def prompt_from_dict(self, query: str, data: dict) -> list[dict]:
        messages = [
            {'role': 'user', 'content': f'Hello, you are a rag assistant who '
             'can help me answer the query from the retrieved '
             'data sources in one sentence. Do not think too much.Please'
             f' answer this question({query}) from {data}:'}
        ]
        return messages

    def _load(self) -> None:
        if not self._llm or not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                dtype=torch.float32,
                device_map="cpu"
            )

    def generate(self, messages: list) -> Any:
        self._load()
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors='pt').to('cpu')

        with torch.no_grad():
            output = self._llm.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=False
            )
        generated_ids = cast(torch.Tensor, output)

        trimmed_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in
            zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(trimmed_ids,
                                           skip_special_tokens=True)[0]
