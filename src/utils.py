import sys
import os
# 强制将项目根目录加入Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hashlib import md5
from typing import List, Dict, Optional
import httpx
from openai import OpenAI
import re
import string
import logging
import numpy as np
import os
from src import config
from dotenv import load_dotenv

load_dotenv()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    计算内容的MD5哈希值作为唯一标识符
    
    Args:
        content: 需要计算哈希的内容字符串
        prefix: 可选的前缀字符串
        
    Returns:
        带前缀的MD5哈希字符串
    """
    return prefix + md5(content.encode()).hexdigest()


class LLMModel:
    """
    LLM模型调用封装类
    支持OpenAI兼容API的模型调用
    """
    
    def __init__(self):
        """
        初始化LLM客户端
        
        Args:
            llm_model: 模型名称
            api_key: API密钥，默认从环境变量OPENAI_API_KEY获取
            base_url: API基础URL，默认从环境变量OPENAI_BASE_URL获取
            max_tokens: 最大生成token数
            temperature: 温度参数，控制随机性
            timeout: 请求超时时间（秒）
        """
        self.api_key = config.api_key
        self.base_url = config.base_url
        
        http_client = httpx.Client(timeout=600.0, trust_env=True)
        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client
        )
        
        self.llm_config = {
            "model": config.llm_model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
    
    def infer(self, messages) -> str:
        """
        执行推理
        
        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
            
        Returns:
            模型生成的文本内容
        """
        response = self.openai_client.chat.completions.create(
            **self.llm_config,
            messages=messages,
        )
        return response.choices[0].message.content




def normalize_answer(s: Optional[str]) -> str:
    """
    标准化答案文本
    
    处理步骤：
    1. 转换为小写
    2. 移除标点符号
    3. 移除冠词(a, an, the)
    4. 规范化空白字符
    
    Args:
        s: 待标准化的字符串
        
    Returns:
        标准化后的字符串
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def setup_logging(
    log_file: str = "logs/app.log",
    level: int = logging.INFO
) -> None:
    """
    配置日志系统
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # 抑制httpx/openai的噪声日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """
    最小-最大归一化
    
    将数组值缩放到[0, 1]范围
    
    Args:
        x: 输入numpy数组
        
    Returns:
        归一化后的数组
    """
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # 处理所有值相同的情况（范围为0）
    if range_val == 0:
        return np.ones_like(x)
    
    return (x - min_val) / range_val


if __name__ == "__main__":
    # 示例：使用compute_mdhash_id生成节点ID
    nodes_dict = {}
    for text in ["apple", "banana", "orange"]:
        node_id = compute_mdhash_id(text, prefix="namespace-")
        nodes_dict[node_id] = {'content': text}
    
    all_hash_ids = list(nodes_dict.keys())
    print("生成的哈希ID:")
    print(all_hash_ids)
    print("\n节点字典:")
    print(nodes_dict)
    
    # 示例：测试normalize_answer
    test_answers = [
        "The Apple Inc.",
        "A banana fruit!",
        "An orange, please?",
        None,
        123
    ]
    print("\n标准化测试结果:")
    for ans in test_answers:
        normalized = normalize_answer(ans)
        print(f"原始: {ans!r:20} -> 标准化: {normalized!r}")
    
    # 示例：测试min_max_normalize
    test_array = np.array([10, 20, 30, 40, 50])
    normalized_array = min_max_normalize(test_array)
    print(f"\n归一化测试:")
    print(f"原始数组: {test_array}")
    print(f"归一化后: {normalized_array}")
    
    # 示例：创建LLM模型实例（需要配置环境变量）
    print("\nLLM模型配置示例:")
    try:
        llm = LLMModel()
        print(f"模型: {llm.llm_config['model']}")
        print(f"最大tokens: {llm.llm_config['max_tokens']}")
        print(f"温度: {llm.llm_config['temperature']}")
    except Exception as e:
        print(f"注意: LLM初始化需要配置OPENAI_API_KEY和OPENAI_BASE_URL环境变量")
        print(f"错误信息: {e}")