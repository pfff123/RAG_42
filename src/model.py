import torch
from data import MinimalSource
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model(BaseModel):
    model_name: str = 'Qwen/Qwen3-0.6B'

    def prompt(self, query: str, data: list[MinimalSource]) -> list[dict]:
        content: list = []
        for i in data:
            content.append(i.get_content())
        messages = [
            {'role': 'user', 'content': f'Hello, you are a rag assistant who '
             'can help me answer the query from the retrieved data sources in'
             f'one sentence.Please answer this question({query}) from {content}: '}
        ]
        return messages

    def prompt_from_dict(self, query: str, data: dict) -> list[dict]:
        messages = [
            {'role': 'user', 'content': f'Hello, you are a rag assistant who '
             'can help me answer the query from the retrieved data sources in '
             f'one sentence.Please answer this question({query}) from {data}: '}
        ]
        return messages

    def generate(self, messages: list) -> str:
        model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map='cpu'
                )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors='pt').to('cpu')
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in
            zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids,
                                          skip_special_tokens=True)[0]
        return response

