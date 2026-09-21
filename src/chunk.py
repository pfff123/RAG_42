import glob
import os
import pickle
from pathlib import Path

import bm25s
import tqdm
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pydantic import BaseModel

from answer import MinimalSource


class Chunk(BaseModel):
    all_chunks: list[MinimalSource]
    cwd: str = os.getcwd()

    @classmethod
    def create(cls, path: str, max_chunk_size: int) -> "Chunk":
        """create a Chunk object and make all_chunks by reading and chunking
        different file types

        Args:
            path (str): the relative path of the raw data

        Returns:
            Chunk: a Chunk object
        """
        cwd: str = os.getcwd()
        dir: str = cwd + path
        if os.path.isdir(dir) is False:
            raise ValueError(f"Error: Your input {path} is not valid!")
        md_files: list = glob.glob(f'{dir}/**/*.md', recursive=True)
        py_files: list = glob.glob(f'{dir}/**/*.py', recursive=True)
        chunk_list: list = []
        all_files: list = md_files + py_files
        for file in tqdm.tqdm(all_files, desc="Chunking files", unit="file"):
            with open(file, 'r') as f:
                content: str = f.read()
            relpath: str = str(Path(file).relative_to(cwd))
            if file in md_files:
                chunk_list.extend(cls.chunk_text(content, relpath,
                                                 max_chunk_size))
            elif file in py_files:
                chunk_list.extend(cls.chunk_py(content, relpath,
                                               max_chunk_size))
            else:
                continue
        return cls(all_chunks=chunk_list)

    def process_chunk(self) -> int:
        """output indexed chunks into a pickle file

        Returns:
            _type_: _description_
        """
        processed: str = os.getcwd() + '/data/processed/indexed_chunks.pkl'
        with open(processed, 'wb') as f:
            pickle.dump(self.all_chunks, f)
        return (0)

    @staticmethod
    def chunk_text(content: str, path: str, maxs: int) -> list[MinimalSource]:
        """split the text(md) files

        Args:
            content (str): f.read
            path (str): file path

        Returns:
            list[MinimalSource]
        """        
        text_chunks: list = []
        size: int = min(maxs, 200)
        text_splitter = RecursiveCharacterTextSplitter(
                                                    chunk_size=size,
                                                    chunk_overlap=0)
        chunks = text_splitter.split_text(content)
        start_index: int = 0
        for i, chunk in enumerate(chunks):
            start_index = content.find(chunk[0:50])
            end_index: int = start_index + len(chunk)
            text_chunks.append(MinimalSource(file_path=path,
                                             first_character_index=start_index,
                                             last_character_index=end_index))
        return text_chunks

    @staticmethod
    def chunk_py(content: str, path: str, maxs: int) -> list[MinimalSource]:
        """split the python files

        Args:
            content (str): f.read
            path (str): file path

        Returns:
            list[MinimalSource]
        """
        py_chunks: list = []
        size: int = min(maxs, 100)
        py_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=size,
                    chunk_overlap=0)
        chunks = py_splitter.split_text(content)
        start_index: int = 0
        for i, chunk in enumerate(chunks):
            start_index = content.find(chunk[0:50])
            end_index: int = start_index + len(chunk)
            py_chunks.append(MinimalSource(file_path=path,
                                           first_character_index=start_index,
                                           last_character_index=end_index))
        return py_chunks


class Retriver(BaseModel):
    @staticmethod
    def find_key(data: dict[str, MinimalSource], values: list[str]) -> list:
        result: list = []
        for k, v in data.items():
            if v in values:
                result.append(k)
        return (result)

    @staticmethod
    def retrieve_k(data: list[MinimalSource],
                   query: str, k: int) -> list[MinimalSource]:
        chunk_text = [i.get_content() for i in data]
        chunk_tokens = bm25s.tokenize(chunk_text)
        retriever = bm25s.BM25()
        retriever.index(chunk_tokens)
        query_tokens = bm25s.tokenize(query)
        results, _ = retriever.retrieve(query_tokens, k=k)
        return [data[idx] for idx in results[0]]
