
import json
import os
import pickle
import re
from .chunk import Chunk, Retriver
from .model import Model
from pydantic import BaseModel
from tqdm import tqdm
from .answer import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .data import MinimalSource


class RagCLI(BaseModel):
    retriever: Retriver | None = None
    input_path: str
    data: list[MinimalSource] = []
    model: Model | None = None

    def index(self, max_chunk_size: int) -> None:
        """Ingest data/raw/ and build the index under data/processed/.

        Args:
            max_chunk_size (int): _description_

        Raises:
            ValueError: when chunk size bigger than 2000
        """
        if max_chunk_size > 2000 or max_chunk_size <= 0:
            raise ValueError("Error: max chunk size not valid!")
        new = Chunk.create(self.input_path, max_chunk_size)
        if new.process_chunk() == 0:
            print('Ingestion complete! Indices saved under data/processed/')

    def index_check(self) -> None:
        """check if already done index step

        Raises:
            FileNotFoundError: _description_
        """
        pkl: str = "/data/processed/indexed_chunks.pkl"
        path: str = os.path.dirname(os.getcwd()) + pkl
        path2: str = os.getcwd() + pkl
        if os.path.isfile(path) is False and os.path.isfile(path2) is False:
            raise FileNotFoundError(f"Error: {path} file not found,"
                                    "missing index step!")
        if not self.data:
            with open(path, 'rb') as f:
                self.data = pickle.load(f)

    def only_search(self, query: str, k: int) -> list[MinimalSource]:
        """ Return the top-k sources for a single query, without print it out

        Args:
            query (str): _description_
            k (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            list[MinimalSource]: _description_
        """
        if not query:
            raise ValueError("Error: Empty query")
        self.index_check()
        if k <= 0 or k > 50:
            raise ValueError("Error: wrong input")
        if not self.retriever:
            self.retriever = Retriver()
            self.retriever.build_index(self.data)
        return self.retriever.retrieve_k(query=query, k=k)

    def search(self, query: str, k: int) -> None:
        """_Return the top-k sources for a single query and print it out

        Args:
            query (str): _description_
            k (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            list[MinimalSource]: _description_
        """
        if not query:
            raise ValueError("Error: Empty query")
        self.index_check()
        if k <= 0 or k > 50:
            raise ValueError("Error: wrong input")
        if not self.retriever:
            self.retriever = Retriver()
            self.retriever.build_index(self.data)
        result: list = self.retriever.retrieve_k(query=query, k=k)
        for i in result:
            i.print()

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        """find questions in dataset path;
           search result from it, output in json at save_dir

        Args:
            dataset_path (str): _description_
            k (int): _description_
            save_directory (str): _description_
        """
        data_path = dataset_path if os.path.isabs(dataset_path) \
            else os.path.join(os.path.dirname(os.getcwd()), dataset_path)
        if not save_directory or not dataset_path:
            raise ValueError("Error: no save_dir or dataset_path!")
        save_directory = save_directory if os.path.isabs(save_directory) \
            else os.path.join(os.path.dirname(os.getcwd()), save_directory)
        if os.path.isfile(data_path) is False:
            raise ValueError("Error: false input dataset_path")
        if (k <= 0 or k >= 50):
            raise ValueError("Error: false input k")
        self.index_check()
        questions: dict = {}
        with open(data_path, 'r') as f:
            questions = json.load(f)
        pure_questions: list = questions['rag_questions']
        re: list = []
        for question in tqdm(pure_questions, desc="Processing the search"):
            q = question['question']
            retrived: list = self.only_search(q, k)
            re.append(MinimalSearchResults(question_id=question['question_id'],
                                           question=q,
                                           retrieved_sources=retrived))
        output = StudentSearchResults(search_results=re, k=k)
        _, filename = os.path.split(data_path)
        processed: str = os.path.join(save_directory, filename)
        with open(processed, 'w') as f:
            json.dump(output.get_dict(), f, indent=2)
        print(f'Saved student_search_results to {processed}')

    def answer(self, query: str, k: int) -> None:
        """Answer a single query using the retrieved context.

        Args:
            query (str): _description_
            k (int): _description_
        """
        self.index_check()
        if query == "":
            raise ValueError("Error: empty query!")
        if k <= 0 or k > 50:
            raise ValueError("Error: invalid input of k!")
        retreived: list[MinimalSource] = self.only_search(query, k)
        if not self.model:
            self.model = Model()
            self.model._load()
        prompt: list = self.model.prompt(query, retreived)
        answer: str = self.model.generate(prompt)
        pattern: str = r"<think>.*?</think>|$"
        trimed: str = re.sub(pattern, "", answer, flags=re.DOTALL).strip()
        print(trimed)

    def only_answer(self, query: str, data: dict) -> MinimalAnswer:
        """Answer a single query using the retrieved context.

        Args:
            query (str): _description_
            k (int): _description_
        """
        self.index_check()
        if query == "":
            raise ValueError("Error: empty query!")
        if len(data) == 0:
            raise ValueError("Error: empty data!")
        if not self.model:
            self.model = Model()
            self.model._load()
        prompt: list = self.model.prompt_from_dict(query, data)
        answer: str = self.model.generate(prompt)
        pattern: str = r"<think>.*?</think>|$"
        k: str = re.sub(pattern, "", answer, flags=re.DOTALL).strip()
        return MinimalAnswer(answer=k)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """Generate answers for a dataset, producing a
            StudentSearchResultsAndAnswer
            JSON file.

        Args
            student_search_results_path (str): _description_
            save_directory (str): _description_
        """
        self.index_check()
        if os.path.isfile(student_search_results_path) is False:
            raise ValueError("Error: student_search_results_path not valid!")
        os.makedirs(save_directory, exist_ok=True)
        with open(student_search_results_path, "r") as f:
            search_results: dict = json.load(f)
        print(f"Loaded {len(search_results['search_results'])} questions.")
        k: int = search_results['k']
        if not self.model:
            self.model = Model()
            self.model._load()
        answers: list[MinimalAnswer] = []
        for result in tqdm(search_results["search_results"],
                           desc="Processing questions"):
            query: str = result['question']
            retrived: dict = result["retrieved_sources"]
            answers.append(self.only_answer(query, retrived))
        output: StudentSearchResultsAndAnswer = StudentSearchResultsAndAnswer(
            search_results=answers, k=k)
        all_answers: list = output.get_answers()
        for index, result in enumerate(search_results["search_results"]):
            result["answer"] = all_answers[index]
        path: str = os.path.join(save_directory,
                                 "StudentSearchResultsAndAnswer.json")
        with open(path, "w") as f:
            json.dump(search_results, f, indent=2)
        print(f'Saved student_search_results_and_answer to {path}')
