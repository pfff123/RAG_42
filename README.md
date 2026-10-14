*This project has been created as part of the 42 curriculum by yafu[yafu@student.codam.nl].*

# RAG Against the Machine

## Description

This project implements a Retrieval-Augmented Generation (RAG) system from scratch: a pipeline that retrieves relevant chunks of text from a document collection and uses them to ground the answers of a language model, reducing hallucination and allowing the model to answer questions about content it wasn't trained on.

The corpus is the [vLLM](https://github.com/vllm-project/vllm) project repository (its `.py` source files and `.md` documentation). Given a natural-language query, the system indexes the corpus into chunks, retrieves the most relevant passages, and feeds them to an LLM to generate a grounded answer.

## Instructions

### Requirements
- Python >= 3.11, < 3.14
- uv for dependency management

### Installation & usage
```bash
make install   # install dependencies
make run       # run the main script
make debug     # run the main script in debug mode (pdb)
make clean     # remove __pycache__, .mypy_cache, etc.
make lint      # flake8 . and mypy . (see Makefile for full flags)
```

## System Architecture

1. **Ingestion** — source files are read from the corpus (`.py` and `.md` files under `data/raw/`).
2. **Chunking** — each file is split into passages using LangChain's text splitters (see Chunking Strategy).
3. **Indexing** — chunks are indexed with BM25 for lexical retrieval (via `bm25s`).
4. **Retrieval** — given a query, the top-k most relevant chunks are retrieved and returned with their source file and character offsets.
5. **Generation** — retrieved chunks are inserted into a prompt and passed to the LLM (Qwen3-0.6B, loaded via HuggingFace `transformers`) to generate the final answer, with output validated/structured using `pydantic`.

```
Query → Retriever (BM25 / bm25s) → Top-k chunks → Prompt template → LLM (Qwen3-0.6B) → Answer (validated via pydantic)
```

## Chunking Strategy

Chunking is done with LangChain's text splitters:
- **`.md` files** — `RecursiveCharacterTextSplitter`
- **`.py` files** — `RecursiveCharacterTextSplitter.from_language` (Python-aware splitting, so chunks don't cut through code structure as bluntly as plain character splitting would)

For both file types, the chunk size is `min(1650, max_chunk_size)`, where `max_chunk_size` is a configurable input parameter — this caps chunks at 1650 characters by default while still allowing a smaller size to be requested. Chunks overlap by 350 characters, so relevant content near a chunk boundary isn't lost from either side.

## Retrieval Method

Retrieval uses **BM25** (via the [`bm25s`](https://pypi.org/project/bm25s/) library), a lexical ranking algorithm that scores each chunk against the query based on term frequency, inverse document frequency (rarer terms weigh more), and document-length normalization (so longer chunks aren't unfairly favored just for containing more words).

The number of chunks retrieved is configurable via `--k` (e.g. `--k 5`, `--k 10`). Chunks are returned ranked by BM25 score, with no additional re-ranking step.

## Performance Analysis

The retriever was evaluated on two separate question sets, using recall@k (whether a relevant chunk appears in the top-k retrieved results):

**Code questions** (99 questions)

| k | Recall@k |
|---|----------|
| 1 | 30.3% |
| 3 | 48.5% |
| 5 | 56.6% |
| 10 | 60.6% |

**Docs questions** (100 questions)

| k | Recall@k |
|---|----------|
| 1 | 62.0% |
| 3 | 77.0% |
| 5 | 85.0% |
| 10 | 88.0% |

Retrieval performs noticeably better on documentation questions than on code questions across every k. This gap likely reflects how BM25 favors lexical/keyword overlap: documentation text tends to phrase questions and answers in similar natural language, while code-related questions often use terms (function names, concepts) that don't literally appear in the corresponding code chunks, making lexical matching less reliable.

## Design Decisions

- Chose BM25 over dense embeddings for retrieval — no GPU available, and BM25 is simpler to implement and reason about.
- Chose Qwen3-0.6B as the generation model — small enough to run on CPU within Codam's hardware constraints.
- Used language-aware chunking (`RecursiveCharacterTextSplitter.from_language` for `.py` files) rather than plain character splitting, to keep code chunks more structurally coherent.
- Used `pydantic` to enforce structured/validated output from the LLM.

## Challenges Faced

- **Disk space constraints on Codam machines** — the default `torch` dependency was too large to install. Solved by installing the CPU-only build of `torch` instead.
- **mypy false positive on `transformers`' `generate()`** — mypy flagged `llm.generate(...)` as "Tensor not callable" due to how `transformers`' type stubs resolve `PreTrainedModel.generate`. Solved by [adding a targeted `# type: ignore[operator]` / a `follow_imports = "skip"` override for `transformers.*` — state whichever you kept].

## Example Usage

```bash
$ uv run python -m src search "What is the goal of this project?" --k 5

Answer:
data/raw/vllm-0.10.1/docs/mkdocs/hooks/url_schemes.py [0:877]
data/raw/vllm-0.10.1/docs/design/metrics.md [23944:25447]
data/raw/vllm-0.10.1/examples/offline_inference/audio_language.py [0:922]
data/raw/vllm-0.10.1/docs/contributing/README.md [9737:11361]
data/raw/vllm-0.10.1/vllm/worker/model_runner.py [62705:63470]
```

A script to test against the evaluation dataset and moulinette:
```bash
#!/usr/bin/env bash
set -e  # stop on first error, like && chaining but cleaner

echo "testing docs dataset..."
uv run python -m src index --max_chunk_size 2000
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000

echo "testing code dataset..."
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    data/datasets/AnsweredQuestions/dataset_code_public.json \
    --k 10 --max_context_length 2000
```

## Resources

**Chunking**
- [The Ultimate Text Chunking Toolkit — 15 Methods with Python Code](https://medium.com/data-science-collective/the-ultimate-text-chunking-toolkit-15-methods-with-python-code-9ef9d8f6a898)

**TF-IDF & BM25**
- [Understanding TF-IDF (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/)
- [What is BM25? Best Matching 25 Algorithm (GeeksforGeeks)](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/)
- [bm25s on PyPI](https://pypi.org/project/bm25s/)
- [TF-IDF & BM25 video explainer](https://www.bilibili.com/video/BV1N7YMegEaf/)

**File handling / tooling**
- [Get All Files in a Directory in Python (Real Python)](https://realpython.com/get-all-files-in-directory-python/)
- [tqdm on PyPI](https://pypi.org/project/tqdm/)

## AI usage

Claude was used to help debug a mypy type-checking error related to `transformers`' `generate()` method signature, to debug a Hugging Face Hub rate-limit warning and a deprecated `torch_dtype` argument, and to help structure and write this README. All pipeline design, chunking/retrieval logic, and code implementation were written by yafu.