import json
from typing import List, Dict, Any, Optional
from src import config

def load_chunks(filepath: str) -> List[Dict]:
    """加载文本块数据，返回列表字典格式"""
    passages = []
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    for idx, raw_str in enumerate(chunks):
        # 按第一个冒号分割，提取索引和文本
        split_idx = raw_str.find(":")
        if split_idx != -1:
            chunk_idx = raw_str[:split_idx]
            chunk_text = raw_str[split_idx+1:]
        else:
            # 兼容异常情况
            chunk_idx = str(idx)
            chunk_text = raw_str

        passages.append({
            "id": int(chunk_idx),
            "text": chunk_text
        })
    return passages


def load_questions(filepath: str) -> List[Dict]:
    """加载问题数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return questions

if __name__ == "__main__":
    chunks = load_chunks(config.chunks_file)
    print(chunks[:5])
    questions = load_questions(config.questions_file)
    print(questions[:5])
