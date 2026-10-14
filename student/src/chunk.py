import glob
import os
import pickle
from pathlib import Path

import bm25s
import tqdm
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict

from .answer import MinimalSource


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
        dir: str = os.path.dirname(cwd) + path
        if os.path.isdir(dir) is False:
            raise ValueError(f"Error: Your input {path} is not valid!")
        md_files: list = glob.glob(f'{dir}/**/*.md', recursive=True)
        py_files: list = glob.glob(f'{dir}/**/*.py', recursive=True)
        chunk_list: list = []
        all_files: list = md_files + py_files
        for file in tqdm.tqdm(all_files, desc="Chunking files", unit="file"):
            with open(file, 'r') as f:
                content: str = f.read()
            relpath: str = str(Path(file).relative_to(os.path.dirname(cwd)))
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
        parent_dir: str = os.path.dirname(os.getcwd())
        out_dir = parent_dir + "/data/processed/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        processed = out_dir + "/indexed_chunks.pkl"
        with open(processed, "wb") as f:
            pickle.dump(self.all_chunks, f)
        return 0

    @staticmethod
    def chunk_to_list(chunks: list[str], content: str,
                      path: str) -> list[MinimalSource]:
        cursor: int = 0
        result: list = []
        for chunk in chunks:
            needle: str = chunk[0:50] if len(chunk) >= 50 else chunk
            start_index: int = content.find(needle, cursor)
            if start_index == -1:
                start_index = content.find(needle)
                if start_index == -1:
                    raise ValueError(f"Couldn't find {chunk[0:50]}")
            end_index: int = start_index + len(chunk)
            result.append(MinimalSource(file_path=path,
                                        first_character_index=start_index,
                                        last_character_index=end_index))
            cursor = start_index + 1
        return result

    @staticmethod
    def chunk_text(content: str, path: str, maxs: int) -> list[MinimalSource]:
        """split the text(md) files

        Args:
            content (str): f.read
            path (str): file path

        Returns:
            list[MinimalSource]
        """
        size: int = min(maxs, 1650)
        text_splitter = RecursiveCharacterTextSplitter(
                                                    chunk_size=size,
                                                    chunk_overlap=350)
        chunks: list = text_splitter.split_text(content)
        return Chunk.chunk_to_list(chunks=chunks, content=content, path=path)

    @staticmethod
    def chunk_py(content: str, path: str, maxs: int) -> list[MinimalSource]:
        """split the python files

        Args:
            content (str): f.read
            path (str): file path

        Returns:
            list[MinimalSource]
        """
        size: int = min(maxs, 1650)
        py_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=size,
                    chunk_overlap=350)
        chunks = py_splitter.split_text(content)
        return Chunk.chunk_to_list(chunks=chunks, content=content, path=path)


class Retriver(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: bm25s.BM25 | None = None
    data: list[MinimalSource] = []

    @staticmethod
    def find_key(data: dict[str, MinimalSource], values: list[str]) -> list:
        result: list = []
        for k, v in data.items():
            if v in values:
                result.append(k)
        return (result)

    def build_index(self, data: list[MinimalSource]) -> None:
        self.data = data
        chunk_text = [i.get_content() for i in data]
        chunk_tokens = bm25s.tokenize(chunk_text)
        self.retriever = bm25s.BM25()
        self.retriever.index(chunk_tokens)

    def retrieve_k(self, query: str, k: int) -> list[MinimalSource]:
        if self.retriever is None or self.data is None:
            raise RuntimeError("Error: Index not build yet")
        query_tokens = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(query_tokens, k=k)
        if scores[0][0] <= 0:
            raise ValueError("Error: strange query or datasets,"
                             "hard to retrieve related sources")
        return [self.data[idx] for idx in results[0]]
