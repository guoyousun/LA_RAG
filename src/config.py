import os
from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# llm:
# Model settings (can override env vars)
llm_model = "qwen3-max"
api_key = os.getenv("QWEN_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
temperature = 0.0
max_tokens = 16384
reasoning_effort = "medium"  # For reasoning models (o1, etc.)

# 数据集名称
dataset_name: str = "2wikimultihop"

# embedding:
# HuggingFace model name or local model path
# We use Qwen3-Embedding-0.6B in our paper
embedding_model = os.path.join(ROOT_DIR, "models", "Qwen3-Embedding-0.6B")
# model: "/path/to/local/Qwen3-Embedding-0.6B"
device = "cuda:0"  # or "cpu"
batch_size = 16


# agent:
max_loops= 15
max_token_budget= 128000
verbose=False


# data:
dataset_path = os.path.join(ROOT_DIR, "dataset", dataset_name)
chunks_file = os.path.join(dataset_path, "chunks.json")
questions_file = os.path.join(dataset_path, "questions.json")
index_dir = os.path.join(dataset_path, "index")

# output:
results_dir = os.path.join(ROOT_DIR, "results")


# eval:
predictions_file = os.path.join(results_dir, "predictions.jsonl") # 预测结果文件路径
output_dir = results_dir # 评估结果输出目录
max_workers = 10 # 并行线程数

if __name__ == "__main__":
    print(chunks_file)
    print(questions_file)
    print(index_dir)