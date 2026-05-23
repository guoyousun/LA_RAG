"""
RAG Agent 使用示例
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.react_agent import create_rag_agent
from src.config import results_dir


def main():
    """主函数"""
    
    print("="*80)
    print("ReAct RAG Agent Demo (using create_agent)")
    print("="*80)
    
    # 创建智能体
    agent = create_rag_agent()
    
    # 测试问题
    test_questions = [
        "When did Lothair Ii's mother die?",
        "Which film was released first, Aas Ka Panchhi or Phoolwari?",
        "What is the place of birth of the performer of song Changed It?"
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}/{len(test_questions)}: {question}")
        print(f"{'='*80}")
        
        # 运行智能体
        result = agent.run(question)
        results.append(result)
        
        # 打印答案
        print(f"\n✅ Final Answer:")
        print(result['answer'])
        print(f"\n📊 Success: {result['success']}")
    
    # 保存结果
    output_file = os.path.join(results_dir, "rag_agent_results.json")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()