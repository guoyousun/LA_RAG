"""
RAG 智能体答案评估模块
支持 LLM 评估和包含匹配评估
"""
import json
import os
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

from src.utils import normalize_answer

logger = logging.getLogger(__name__)


class Evaluator:
    """
    RAG 智能体答案评估器
    
    支持两种评估方式：
    1. LLM 评估：使用大模型判断答案是否正确
    2. Contain 评估：检查预测答案是否包含标准答案的关键信息
    """
    
    def __init__(self, llm_model, predictions_path: str):
        """
        初始化评估器
        
        Args:
            llm_model: LLM 模型实例（用于 LLM 评估）
            predictions_path: 预测结果文件路径
        """
        self.llm_model = llm_model
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()
        
        logger.info(f"Loaded {len(self.prediction_results)} predictions from {predictions_path}")
    
    def load_predictions(self) -> List[Dict]:
        """加载预测结果"""
        try:
            with open(self.predictions_path, 'r', encoding='utf-8') as f:
                prediction_results = json.load(f)
            return prediction_results
        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")
            raise
    
    def calculate_llm_accuracy(self, pred_answer: str, gold_answer: str) -> float:
        """
        使用 LLM 评估答案准确性
        
        Args:
            pred_answer: 预测答案
            gold_answer: 标准答案
            
        Returns:
            1.0 表示正确，0.0 表示错误
        """
        system_prompt = """You are an expert evaluator for question answering systems. 
Your task is to determine if the predicted answer is correct by comparing it with the gold (standard) answer."""
        
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.

Generated answer: {pred_answer}
Gold answer: {gold_answer}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Note:
- Minor wording differences are acceptable
- The answer can be more detailed than the gold answer
- Focus on factual correctness, not exact matching

Respond with ONLY 'correct' or 'incorrect'.
Response:"""
        
        try:
            response = self.llm_model.infer([
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ])
            
            if response.strip().lower() == "correct":
                return 1.0
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            return 0.0
    
    def calculate_contain(self, pred_answer: str, gold_answer: str) -> int:
        """
        检查预测答案是否包含标准答案
        
        Args:
            pred_answer: 预测答案
            gold_answer: 标准答案
            
        Returns:
            1 表示包含，0 表示不包含
        """
        # 检查空值
        if not pred_answer or not isinstance(pred_answer, str) or pred_answer.strip() == "":
            return 0
        
        if not gold_answer or not isinstance(gold_answer, str) or gold_answer.strip() == "":
            return 0
        
        # 标准化后检查包含关系
        s1 = normalize_answer(pred_answer)
        s2 = normalize_answer(gold_answer)
        
        if s2 in s1:
            return 1
        else:
            return 0
    
    def evaluate_single_sample(self, idx: int, prediction: Dict) -> Tuple[int, float, int]:
        """
        评估单个样本
        
        Args:
            idx: 样本索引
            prediction: 预测结果字典
            
        Returns:
            (idx, llm_accuracy, contain_accuracy)
        """
        pred_answer = prediction.get("pred_answer", "")
        gold_answer = prediction.get("gold_answer", "")
        
        # LLM 评估
        llm_acc = self.calculate_llm_accuracy(pred_answer, gold_answer)
        
        # Contain 评估
        contain_acc = self.calculate_contain(pred_answer, gold_answer)
        
        return idx, llm_acc, contain_acc
    
    def evaluate(self, max_workers: int = 10) -> Tuple[float, float]:
        """
        评估所有预测结果
        
        Args:
            max_workers: 并行线程数
            
        Returns:
            (llm_accuracy, contain_accuracy)
        """
        print(f"\n{'='*80}")
        print(f"Starting Evaluation")
        print(f"{'='*80}")
        print(f"Total samples: {len(self.prediction_results)}")
        print(f"Max workers: {max_workers}")
        print(f"{'='*80}\n")
        
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.evaluate_single_sample, idx, pred): idx 
                for idx, pred in enumerate(self.prediction_results)
            }
            
            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            
            # 使用进度条显示评估进度
            pbar = tqdm(
                total=len(futures), 
                desc="Evaluating samples", 
                unit="sample"
            )
            
            for future in as_completed(futures):
                try:
                    idx, llm_acc, contain_acc = future.result()
                    
                    # 保存分数
                    llm_scores[idx] = llm_acc
                    contain_scores[idx] = contain_acc
                    
                    # 更新预测结果
                    self.prediction_results[idx]["llm_accuracy"] = llm_acc
                    self.prediction_results[idx]["contain_accuracy"] = contain_acc
                    
                    # 累计分数
                    total_llm_score += llm_acc
                    total_contain_score += contain_acc
                    completed += 1
                    
                    # 计算当前准确率
                    current_llm_acc = total_llm_score / completed
                    current_contain_acc = total_contain_score / completed
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'LLM_Acc': f'{current_llm_acc:.3f}',
                        'Contain_Acc': f'{current_contain_acc:.3f}'
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error evaluating sample: {e}")
                    pbar.update(1)
            
            pbar.close()
        
        # 计算最终准确率
        llm_accuracy = sum(llm_scores) / len(llm_scores) if llm_scores else 0.0
        contain_accuracy = sum(contain_scores) / len(contain_scores) if contain_scores else 0.0
        
        # 打印评估结果
        print(f"\n{'='*80}")
        print("Evaluation Results:")
        print(f"{'='*80}")
        print(f"  Total samples: {len(self.prediction_results)}")
        print(f"  LLM Accuracy: {llm_accuracy:.4f} ({sum(llm_scores):.0f}/{len(llm_scores)})")
        print(f"  Contain Accuracy: {contain_accuracy:.4f} ({sum(contain_scores):.0f}/{len(contain_scores)})")
        print(f"{'='*80}\n")
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  LLM Accuracy: {llm_accuracy:.4f} ({sum(llm_scores)}/{len(llm_scores)})")
        logger.info(f"  Contain Accuracy: {contain_accuracy:.4f} ({sum(contain_scores)}/{len(contain_scores)})")
        
        # 保存更新后的预测结果（包含评估分数）
        with open(self.predictions_path, "w", encoding="utf-8") as f:
            json.dump(self.prediction_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存评估汇总结果
        eval_summary_path = os.path.join(
            os.path.dirname(self.predictions_path), 
            "evaluation_results.json"
        )
        
        eval_summary = {
            "llm_accuracy": llm_accuracy,
            "contain_accuracy": contain_accuracy,
            "total_samples": len(self.prediction_results),
            "correct_llm": int(sum(llm_scores)),
            "correct_contain": int(sum(contain_scores))
        }
        
        with open(eval_summary_path, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to:")
        print(f"  - Predictions with scores: {self.predictions_path}")
        print(f"  - Evaluation summary: {eval_summary_path}\n")
        
        return llm_accuracy, contain_accuracy


def evaluate_predictions(
    predictions_path: str,
    llm_model=None,
    max_workers: int = 10
) -> Tuple[float, float]:
    """
    便捷函数：评估预测结果
    
    Args:
        predictions_path: 预测结果文件路径
        llm_model: LLM 模型实例（如果为 None，则只进行 contain 评估）
        max_workers: 并行线程数
        
    Returns:
        (llm_accuracy, contain_accuracy)
    """
    evaluator = Evaluator(llm_model, predictions_path)
    return evaluator.evaluate(max_workers)