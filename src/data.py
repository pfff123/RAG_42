import glob
import json
import os
import pathlib
import uuid
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

    @classmethod
    def create(cls, data) -> "MinimalSource":
        return cls(file_path=data['file_path'],
                   first_character_index=data['first_character_index'],
                   last_character_index=data['last_character_index'])

    @model_validator(mode="after")
    def checkvalid(self) -> "MinimalSource":
        if self.first_character_index >= self.last_character_index:
            raise ValueError(f"Error: first_character_index"
                             f"{self.first_character_index}should be "
                             f"smaller than last_character_index"
                             f"{self.last_character_index}")
        return self

    def get_content(self) -> str:
        try:
            with open(self.file_path, 'r') as f:
                content: str = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {self.file_path} not found")
        return(content[self.first_character_index: self.last_character_index])

    def print(self):
        print(f'{self.file_path} [{self.first_character_index}:{self.last_character_index}]')
    
    def get_dict(self) -> dict:
        return {"file_path": self.file_path,
                "first_character_index": self.first_character_index,
                "last_character_index": self.last_character_index
                }

class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

    @classmethod
    def create(cls, data: dict) -> "UnansweredQuestion":
        return cls(**data)


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str
    difficulty: str
    is_valid: bool

    @classmethod
    def create_s(cls, data: dict,
                 sources: list[MinimalSource]) -> "AnsweredQuestion":
        data_copy = {**data, "sources": sources}
        return cls(**data_copy)


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

    @classmethod
    def create_dataset(cls, un_path: str, an_path: str):
        rag_data: list = []
        rag_data.extend(cls.create_unanswered(un_path))
        rag_data.extend(cls.create_answered(an_path))
        return cls(rag_questions=rag_data)

    @staticmethod
    def create_unanswered(path: str) -> list[UnansweredQuestion]:
        questions: list = []
        unanswer_path: str = os.getcwd() + path
        unanswer_files: list = os.listdir(unanswer_path)
        for file in unanswer_files:
            path_f: str = unanswer_path + file
            with open(path_f, "r") as f:
                d = json.load(f)
                for i in d['rag_questions']:
                    questions.append(UnansweredQuestion.create(i))
        return questions
    
    @staticmethod
    def create_answered(path: str):
        questions: list = []
        answer_path: str = os.getcwd() + path
        answer_files: list = os.listdir(answer_path)
        for file in answer_files:
            path_f: str = answer_path + file
            with open(path_f, "r") as f:
                d = json.load(f)
                for i in d['rag_questions']:
                    sources: list = []
                    for source in i['sources']:
                        sources.append(MinimalSource.create(source))
                    questions.append(
                        AnsweredQuestion.create_s(i, sources))
        return questions




def main():
    chunk: Chunk = Chunk.create('/data/raw/vllm-0.10.1')
    un_path = '/data/datasets/UnansweredQuestions/'
    an_path = '/data/datasets/AnsweredQuestions/'
    ragdata: RagDataset = RagDataset.create_dataset(un_path, an_path)
    chunk_text = [i.get_content() for i in chunk.all_chunks]
    chunk_tokens = bm25s.tokenize(chunk_text)
    retriever = bm25s.BM25()
    retriever.index(chunk_tokens)
    query = "The `--enable-auto-tool-choice` flag is mandatory to enable automatic function calling in vLLM, as it tells vLLM that you want to enable the model to generate its own tool calls when it deems appropriate."
    query_tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(query_tokens, k=2, corpus=chunk_text)
    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(f"Rank {i+1} (score: {score:.2f}): {doc}")
if __name__=="__main__":
    main()
    