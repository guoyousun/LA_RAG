"""
完整的 RAG 智能体问答系统
整合知识图谱构建、检索和 ReAct 智能体
"""
import sys
import os
import json
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ROOT_DIR, dataset_name, results_dir
from src.data_loader import load_questions
from src.kg_builder import KnowledgeGraphBuilder
from src.kg_retriever import KnowledgeGraphRetriever
from src.utils import LLMModel
from agent.react_agent import create_rag_agent
from src.evaluate import evaluate_predictions


class RAGAgentSystem:
    """
    完整的 RAG 智能体问答系统
    
    工作流程：
    1. 构建知识图谱（可选，如果已构建可跳过）
    2. 初始化检索器
    3. 创建 ReAct 智能体
    4. 执行问答
    5. 评估答案质量
    """
    
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None,
        build_kg: bool = True,
        chunks_file: str = None
    ):
        """
        初始化 RAG 智能体系统
        
        Args:
            neo4j_uri: Neo4j 连接 URI
            neo4j_username: Neo4j 用户名
            neo4j_password: Neo4j 密码
            build_kg: 是否重新构建知识图谱
            chunks_file: chunks 文件路径（仅在 build_kg=True 时需要）
        """
        # 配置参数
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "12345678")
        
        self.embedding_model_path = os.path.join(ROOT_DIR, "models", "Qwen3-Embedding-0.6B")
        self.chunks_file = chunks_file or os.path.join(ROOT_DIR, "dataset", dataset_name, "chunks.json")
        
        # 组件
        self.kg_retriever = None
        self.rag_agent = None
        self.llm_model = None
        
        # 步骤 1: 构建知识图谱（如果需要）
        if build_kg:
            self._build_knowledge_graph()
        
        # 步骤 2: 初始化检索器
        self._init_retriever()
        
        # 步骤 3: 创建 ReAct 智能体
        self._init_agent()
        
        # 步骤 4: 初始化 LLM 模型（用于评估）
        self._init_llm()
    
    def _build_knowledge_graph(self):
        """构建知识图谱"""
        print("\n" + "="*80)
        print("[Step 1] Building Knowledge Graph...")
        print("="*80)
        
        try:
            with KnowledgeGraphBuilder(
                neo4j_uri=self.neo4j_uri,
                neo4j_username=self.neo4j_username,
                neo4j_password=self.neo4j_password,
                use_nlp=True,  # 如果没有安装 spaCy，设为 False
                embedding_model_path=self.embedding_model_path
            ) as builder:
                builder.build_from_chunks_file(self.chunks_file, batch_size=50)
            
            print("\n[Step 1] Knowledge Graph construction completed!\n")
        except Exception as e:
            print(f"\n[Warning] Failed to build knowledge graph: {e}")
            print("Continuing without KG building (assuming KG already exists)...\n")
    
    def _init_retriever(self):
        """初始化知识图谱检索器"""
        print("="*80)
        print("[Step 2] Initializing Knowledge Graph Retriever...")
        print("="*80)
        
        self.kg_retriever = KnowledgeGraphRetriever(
            neo4j_uri=self.neo4j_uri,
            neo4j_username=self.neo4j_username,
            neo4j_password=self.neo4j_password,
            embedding_model_path=self.embedding_model_path
        )
        
        print("[Step 2] Knowledge Graph Retriever initialized successfully\n")
    
    def _init_agent(self):
        """初始化 ReAct 智能体"""
        print("="*80)
        print("[Step 3] Creating ReAct RAG Agent...")
        print("="*80)
        
        self.rag_agent = create_rag_agent()
        
        print("[Step 3] ReAct RAG Agent created successfully\n")
    
    def _init_llm(self):
        """初始化 LLM 模型（用于评估）"""
        print("="*80)
        print("[Step 4] Initializing LLM Model for Evaluation...")
        print("="*80)
        
        self.llm_model = LLMModel()
        
        print("[Step 4] LLM Model initialized successfully\n")
    
    def answer_question_with_gold(
        self, 
        question_data: Dict[str, Any], 
        use_kg_context: bool = True
    ) -> Dict[str, Any]:
        """
        回答单个问题（包含标准答案）
        
        Args:
            question_data: 问题数据字典（包含 id, question, answer 等字段）
            use_kg_context: 是否先使用 KG 检索增强上下文
            
        Returns:
            包含问题、标准答案、预测答案和相关信息的字典
        """
        question_id = question_data.get('id', 'unknown')
        question_text = question_data.get('question', '')
        gold_answer = question_data.get('answer', '')
        
        print(f"\n{'='*80}")
        print(f"Question ID: {question_id}")
        print(f"Question: {question_text}")
        print(f"Gold Answer: {gold_answer}")
        print(f"{'='*80}\n")
        
        result = {
            'id': question_id,
            'question': question_text,
            'gold_answer': gold_answer,
            'pred_answer': '',
            'question_type': question_data.get('question_type', ''),
            'evidence': question_data.get('evidence', []),
            'kg_context': None,
            'agent_result': None,
            'success': False
        }
        
        try:
            # 可选：先进行 KG 检索获取上下文
            if use_kg_context and self.kg_retriever:
                print("[Phase 1] Retrieving context from Knowledge Graph...")
                
                # 确保使用正确的参数名：top_k_sentences
                kg_results = self.kg_retriever.retrieve_by_query(
                    query=question_text,
                    top_k_sentences=5,      
                    max_hops=2              
                )
                
                kg_context = self.kg_retriever.format_for_rag(kg_results)
                result['kg_context'] = kg_context
                
                print(f"[Phase 1] Retrieved KG context ({len(kg_context)} chars)\n")
            else:
                kg_context = ""
            
            # 使用 ReAct 智能体回答问题
            print("[Phase 2] Running ReAct Agent...")
            agent_result = self.rag_agent.run(question_text)
            result['agent_result'] = agent_result
            result['pred_answer'] = agent_result.get('answer', '')
            result['success'] = agent_result.get('success', False)
            
            print(f"\n✅ Predicted Answer:")
            print(result['pred_answer'])
            
            return result
            
        except Exception as e:
            error_msg = f"Error answering question: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            
            result['pred_answer'] = f"Sorry, I encountered an error: {str(e)}"
            result['success'] = False
            return result
    
    def answer_all_questions(
        self, 
        questions_file: str,
        use_kg_context: bool = True,
        output_file: str = None,
        max_questions: int = None
    ) -> List[Dict[str, Any]]:
        """
        回答所有问题
        
        Args:
            questions_file: 问题文件路径
            use_kg_context: 是否使用 KG 上下文
            output_file: 输出文件路径（可选）
            max_questions: 最大处理问题数量（None 表示全部）
            
        Returns:
            结果列表
        """
        print(f"\n{'='*80}")
        print(f"Loading questions from: {questions_file}")
        print(f"{'='*80}\n")
        
        # 加载问题
        questions = load_questions(questions_file)
        
        if max_questions:
            questions = questions[:max_questions]
            print(f"Processing first {max_questions} questions out of {len(questions)} total\n")
        else:
            print(f"Processing all {len(questions)} questions\n")
        
        results = []
        
        for i, question_data in enumerate(questions, 1):
            print(f"\n{'─'*80}")
            print(f"Progress: {i}/{len(questions)}")
            print(f"{'─'*80}")
            
            result = self.answer_question_with_gold(question_data, use_kg_context)
            results.append(result)
            
            # 每处理 5 个问题保存一次中间结果
            if i % 5 == 0 and output_file:
                checkpoint_file = output_file.replace('.json', f'_checkpoint_{i}.json')
                self._save_results(results, checkpoint_file)
                print(f"\n[Checkpoint] Saved intermediate results at question {i}")
        
        # 保存最终结果
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def evaluate_predictions(
        self, 
        predictions_file: str,
        max_workers: int = 10
    ):
        """
        评估预测结果
        
        Args:
            predictions_file: 预测结果文件路径
            max_workers: 并行线程数
        """
        if not self.llm_model:
            print("[Warning] LLM model not initialized, skipping LLM evaluation")
            return None, None
        
        print(f"\n{'='*80}")
        print("[Step 5] Evaluating Predictions...")
        print(f"{'='*80}\n")
        
        llm_accuracy, contain_accuracy = evaluate_predictions(
            predictions_path=predictions_file,
            llm_model=self.llm_model,
            max_workers=max_workers
        )
        
        return llm_accuracy, contain_accuracy
    
    def _save_results(self, results: List[Dict], output_file: str):
        """保存结果到文件"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n{'='*80}")
            print(f"Results saved to: {output_file}")
            print(f"Total answers: {len(results)}")
            print(f"Successful: {sum(1 for r in results if r['success'])}")
            print(f"Failed: {sum(1 for r in results if not r['success'])}")
            print(f"{'='*80}")
        except Exception as e:
            print(f"[Warning] Failed to save results: {e}")
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {
            'neo4j_uri': self.neo4j_uri,
            'embedding_model': self.embedding_model_path,
            'kg_retriever_initialized': self.kg_retriever is not None,
            'rag_agent_initialized': self.rag_agent is not None,
            'llm_model_initialized': self.llm_model is not None,
            'conversation_count': len(self.rag_agent.conversation_history) if self.rag_agent else 0
        }
        return info
    
    def close(self):
        """关闭系统资源"""
        if self.kg_retriever:
            self.kg_retriever.close()
        print("[System] Resources closed")


def main():
    """主函数 - 演示完整的 RAG 智能体问答系统"""
    
    print("="*80)
    print("Complete RAG Agent System - Batch Question Answering & Evaluation")
    print("="*80)
    
    # 配置
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
    
    BUILD_KG = False  # 设置为 True 如果需要重新构建知识图谱
    CHUNKS_FILE = os.path.join(ROOT_DIR, "dataset", dataset_name, "chunks.json")
    QUESTIONS_FILE = os.path.join(ROOT_DIR, "dataset", dataset_name, "questions.json")
    
    # 初始化系统
    system = RAGAgentSystem(
        neo4j_uri=NEO4J_URI,
        neo4j_username=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
        build_kg=BUILD_KG,
        chunks_file=CHUNKS_FILE
    )
    
    # 打印系统信息
    print("\n" + "="*80)
    print("System Information:")
    print("="*80)
    sys_info = system.get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    # 批量回答所有问题
    output_file = os.path.join(results_dir, "rag_agent_predictions.json")
    
    results = system.answer_all_questions(
        questions_file=QUESTIONS_FILE,
        use_kg_context=True,
        output_file=output_file,
        max_questions=20  # 设置为具体数字可以只测试前 N 个问题，如 10
    )
    
    # 打印问答总结
    print(f"\n{'='*80}")
    print("Question Answering Summary:")
    print(f"{'='*80}")
    print(f"Total questions: {len(results)}")
    print(f"Successful answers: {sum(1 for r in results if r['success'])}")
    print(f"Failed answers: {sum(1 for r in results if not r['success'])}")
    
    # 评估预测结果
    llm_accuracy, contain_accuracy = system.evaluate_predictions(
        predictions_file=output_file,
        max_workers=10
    )
    
    # 打印最终总结
    print(f"\n{'='*80}")
    print("Final Summary:")
    print(f"{'='*80}")
    if llm_accuracy is not None:
        print(f"LLM Accuracy: {llm_accuracy:.4f}")
    if contain_accuracy is not None:
        print(f"Contain Accuracy: {contain_accuracy:.4f}")
    
    # 关闭系统
    system.close()
    
    print(f"\n{'='*80}")
    print("Batch question answering and evaluation completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()