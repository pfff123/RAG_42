from .cli import RagCLI
import fire

if __name__ == "__main__":
    try:
        fire.Fire(RagCLI(input_path="/data/raw/vllm-0.10.1",
                         retriever=None, data=[]))
    except Exception as e:
        print(e)
