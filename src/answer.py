from pydantic import BaseModel

from data import MinimalSource


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]

    def get_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "retrieved_sources": [i.get_dict() for i in self.retrieved_sources]
        }


class MinimalAnswer(MinimalSearchResults):
    answer: str

    def print(self) -> None:
        print(self.answer)

    def get(self) -> str:
        return self.answer


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int

    def get_dict(self) -> dict:
        return {
            "search_results": [i.get_dict() for i in self.search_results],
            "k": self.k
        }


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int

    def get_answers(self) -> list[str]:
        return [i.get() for i in self.search_results]
