
import json
import os
import pickle
import re
from chunk import Chunk, Retriver
import fire
from model import Model
from pydantic import BaseModel
from tqdm import tqdm
from answer import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from data import MinimalSource


class RagCLI(BaseModel):
    input_path: str

    def index(self, max_chunk_size: int) -> int:
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
            return (1)
        return (0)

    def index_check(self) -> None:
        path: str = "data/processed/indexed_chunks.pkl"
        if os.path.isfile(path) is False:
            raise FileNotFoundError(f"Error: {path} file not found,"
                                    "missing index step!")

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
        self.index_check()
        if query == "" or k <= 0 or k > 50:
            raise ValueError("Error: wrong input")
        with open("data/processed/indexed_chunks.pkl", 'rb') as f:
            chunk = pickle.load(f)
        retriver = Retriver.retrieve_k(data=chunk, query=query, k=k)
        return retriver

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """_Return the top-k sources for a single query and print it out

        Args:
            query (str): _description_
            k (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            list[MinimalSource]: _description_
        """
        self.index_check()
        if query == "" or k <= 0 or k > 50:
            raise ValueError("Error: wrong input")
        with open("data/processed/indexed_chunks.pkl", 'rb') as f:
            chunk = pickle.load(f)
        retriver = Retriver.retrieve_k(data=chunk, query=query, k=k)
        for i in retriver:
            i.print()
        return retriver

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> StudentSearchResults:
        """find questions in dataset path;
           search result from it, output in json at save_dir

        Args:
            dataset_path (str): _description_
            k (int): _description_
            save_directory (str): _description_
        """
        self.index_check()
        # dataset_path = 'data/datasets/UnansweredQuestions/dataset_docs_public.json'
        questions: dict = {}
        if os.path.isfile(dataset_path) is False or (k <= 0 or k >= 50):
            raise ValueError("Error: false input dataset_path or k")
        os.makedirs(save_directory, exist_ok=True)
        with open(dataset_path, 'r') as f:
            questions = json.load(f)
        pure_questions: list = [question['question'] for question in questions['rag_questions']]
        items: list = []
        for index, q in enumerate(tqdm(pure_questions, )):
            retrived: list = self.only_search(q, k)
            items.append(MinimalSearchResults(question_id=f'q{index+1}',
                                              question=q,
                                              retrieved_sources=retrived))
        output = StudentSearchResults(search_results=items, k=k)
        _, filename = os.path.split(dataset_path)
        processed: str = save_directory + f'/{filename}'
        with open(processed, 'w') as f:
            json.dump(output.get_dict(), f, indent=2)
        print(f'Saved student_search_results to {processed}')
        # return (output)

    def answer(self, query: str, k: int) -> MinimalAnswer:
        """Answer a single query using the retrieved context.

        Args:
            query (str): _description_
            k (int): _description_
        """
        from data import MinimalAnswer
        self.index_check()
        if query == "":
            raise ValueError("Error: empty query!")
        if k <= 0 or k > 50:
            raise ValueError("Error: invalid input of k!")
        retreived: list[MinimalSource] = self.only_search(query, k)
        model = Model()
        prompt: list = model.prompt(query, retreived)
        answer: str = model.generate(prompt)
        pattern: str = r"<think>.*?</think>"
        trimed: str = re.sub(pattern, "", answer, flags=re.DOTALL).strip()
        result = MinimalAnswer(answer=trimed)
        print(result.print())
        return (result)

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
        model = Model()
        prompt: list = model.prompt_from_dict(query, data)
        answer: str = model.generate(prompt)
        pattern: str = r"<think>.*?</think>"
        result = MinimalAnswer(answer=re.sub(pattern, "",
                                             answer, flags=re.DOTALL).strip())
        return (result)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> StudentSearchResultsAndAnswer:
        """Generate answers for a dataset, producing a
            StudentSearchResultsAndAnswer
            JSON file.

        Args
            student_search_results_path (str): _description_
            save_directory (str): _description_
            # uv run python3 src/parse.py answer_dataset -student_search_results_path "data/output/search_results/UnansweredQuestions/StudentSearchResults.json" -save_directory "data/output/search_results_and_answer"
        """
        self.index_check()
        if os.path.isfile(student_search_results_path) is False:
            raise ValueError("Error: student_search_results_path not valid!")
        os.makedirs(save_directory, exist_ok=False)
        with open(student_search_results_path, "r") as f:
            search_results: dict = json.load(f)
        k: int = search_results['k']
        answers: list[MinimalAnswer] = []
        for result in search_results["search_results"]:
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
        return output

#     def evaluate(student_search_results_path: str, dataset_path: str):
#         """Report your own recall@k against a ground-truth dataset, for your own testing.
# ./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
#         Args:
#             student_search_results_path (str): _description_
#             dataset_path (str): _description_
#         """


if __name__ == "__main__":
    fire.Fire(RagCLI(input_path="/data/raw/vllm-0.10.1"))
