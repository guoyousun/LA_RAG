
"""
ReAct RAG Agent using langchain.agents.create_agent
基于最新 create_agent API 的 RAG 智能体
"""
import sys
import os
import re
import json
from typing import List, Dict, Any, Optional
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import llm_model, api_key, base_url, max_loops, temperature, max_tokens
from agent.agent_tools import (
    decide_retrieval_strategy,
    vector_search,
    kg_multi_hop_search,
    evaluate_answer_quality,
    combine_retrieval_results,
    finalize_answer
)


class ReActRAGAgent:
    """
    基于 create_agent API 的 ReAct RAG 智能体
    """
    
    def __init__(self):
        """初始化智能体"""
        self.llm = self._create_llm()
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.conversation_history = []
        
    def _create_llm(self):
        """创建 LLM 实例"""
        llm = ChatTongyi(
            model=llm_model,
            dashscope_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url
        )
        print(f"[ReActRAGAgent] Initialized LLM: {llm_model}")
        return llm
    
    def _create_tools(self):
        """创建工具列表"""
        from agent.agent_tools import generate_answer_from_context
        
        tools = [
            decide_retrieval_strategy,
            vector_search,
            kg_multi_hop_search,
            generate_answer_from_context,
            evaluate_answer_quality,
            combine_retrieval_results,
            finalize_answer
        ]
        print(f"[ReActRAGAgent] Created {len(tools)} tools")
        return tools
    
    def _create_agent(self):
        """使用 create_agent 创建智能体"""
        
        # 定义系统提示
        system_prompt = """You are an advanced RAG (Retrieval-Augmented Generation) agent that answers questions by intelligently retrieving information from multiple sources.

AVAILABLE TOOLS:
1. decide_retrieval_strategy: Analyze the query and decide retrieval strategy
2. vector_search: Search for semantically similar documents in vector database
3. kg_multi_hop_search: Perform multi-hop retrieval in knowledge graph
4. generate_answer_from_context: Generate answer using LLM based on retrieved context and question
5. evaluate_answer_quality: Evaluate answer quality and determine if more retrieval is needed
6. combine_retrieval_results: Merge results from vector and KG searches
7. finalize_answer: Validate and finalize the answer

WORKFLOW:
1. First, use decide_retrieval_strategy to analyze the question
2. Based on the strategy, use appropriate search tools (vector_search and/or kg_multi_hop_search)
3. For multi-hop questions, you MUST perform multiple searches
4. Use combine_retrieval_results to merge information from different sources (if using both)
5. Use generate_answer_from_context to generate an answer based on the retrieved context
6. Use evaluate_answer_quality to check if your answer is good enough
7. If quality is low, perform additional retrieval and regenerate
8. Finally, use finalize_answer to validate and produce the final answer

IMPORTANT RULES:
- Think step by step
- You can use tools multiple times if needed (up to 15 iterations)
- ALWAYS use generate_answer_from_context to generate answers - don't try to answer directly
- Always evaluate answer quality before finalizing
- If you cannot find sufficient information after multiple attempts, state that clearly

When you have a satisfactory answer, use finalize_answer to confirm it.
"""
        
        # 使用 create_agent 创建智能体
        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
        
        print("[ReActRAGAgent] Agent created successfully using create_agent")
        return agent
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        运行智能体回答问题
        
        Args:
            question: 用户问题
            
        Returns:
            包含答案和中间步骤的字典
        """
        print(f"\n{'='*80}")
        print(f"[ReActRAGAgent] Processing question: {question}")
        print(f"{'='*80}\n")
        
        try:
            # 构建消息列表
            messages = [
                HumanMessage(content=question)
            ]
            
            # 调用智能体
            result = self.agent.invoke({
                "messages": messages
            })
            
            # 提取答案
            answer = self._extract_final_answer(result)
            
            # 构建响应
            response = {
                'question': question,
                'answer': answer,
                'raw_result': result,
                'success': True
            }
            
            # 保存对话历史
            self.conversation_history.append({
                'question': question,
                'answer': answer
            })
            
            print(f"\n{'='*80}")
            print(f"[ReActRAGAgent] Answer generated successfully")
            print(f"{'='*80}\n")
            
            return response
            
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            print(f"[ReActRAGAgent] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                'question': question,
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'success': False,
                'error': str(e)
            }
    
    def _extract_final_answer(self, result: Dict) -> str:
        """从结果中提取最终答案"""
        # 尝试从 messages 中提取最后一条 AI 消息
        if 'messages' in result:
            messages = result['messages']
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    
                    # 尝试提取 Final Answer
                    patterns = [
                        r'Final Answer:\s*(.+)',
                        r'FINAL ANSWER:\s*(.+)',
                        r'final answer:\s*(.+)'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                        if match:
                            return match.group(1).strip()
                    
                    # 如果没有找到 Final Answer，返回整个内容
                    return content.strip()
        
        # 如果无法提取，返回原始结果
        return str(result)
    
    def get_conversation_summary(self) -> str:
        """获取对话摘要"""
        if not self.conversation_history:
            return "No conversations yet."
        
        summary = f"Total conversations: {len(self.conversation_history)}\n\n"
        for i, conv in enumerate(self.conversation_history[-5:], 1):
            summary += f"{i}. Q: {conv['question'][:80]}...\n"
            summary += f"   A: {conv['answer'][:100]}...\n\n"
        
        return summary
    
    def reset(self):
        """重置智能体状态"""
        self.conversation_history = []
        print("[ReActRAGAgent] Agent state reset")


def create_rag_agent() -> ReActRAGAgent:
    """工厂函数：创建 RAG 智能体实例"""
    return ReActRAGAgent()


if __name__ == "__main__":
    # 测试智能体
    agent = create_rag_agent()
    
    # 测试问题
    test_questions = [
        "Who was married to Lothair II?",
        "Where was Janis Joplin born?"
    ]
    
    for question in test_questions:
        result = agent.run(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print("-" * 80)

# """
# ReAct RAG Agent using langchain.agents.create_agent
# Optimized for better reasoning and reduced tool complexity
# """
# import sys
# import os
# import re
# import json
# from typing import List, Dict, Any, Optional
# from langchain.agents import create_agent
# from langchain_community.chat_models import ChatTongyi
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.config import llm_model, api_key, base_url, max_loops, temperature, max_tokens
# from agent.agent_tools import (
#     vector_search,
#     kg_multi_hop_search
# )


# class ReActRAGAgent:
#     """
#     Optimized ReAct RAG Agent
#     """
    
#     def __init__(self):
#         """初始化智能体"""
#         self.llm = self._create_llm()
#         self.tools = self._create_tools()
#         self.agent = self._create_agent()
#         self.conversation_history = []
        
#     def _create_llm(self):
#         """创建 LLM 实例"""
#         llm = ChatTongyi(
#             model="qwen3-max",
#             dashscope_api_key=api_key,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             base_url=base_url
#         )
#         print(f"[ReActRAGAgent] Initialized LLM: qwen3-max")
#         return llm
    
#     def _create_tools(self):
#         """创建精简后的工具列表"""
#         tools = [
#             vector_search,
#             kg_multi_hop_search
#         ]
#         print(f"[ReActRAGAgent] Created {len(tools)} optimized tools")
#         return tools
    
#     def _create_agent(self):
#         """使用 create_agent 创建智能体"""
        
#         # 定义增强版系统提示，强调检索策略和失败处理
#         system_prompt = """You are an advanced RAG (Retrieval-Augmented Generation) agent. 
# Your goal is to answer questions accurately by retrieving information from Vector Database and Knowledge Graph.

# AVAILABLE TOOLS:
# 1. vector_search: Search for semantic matches in documents. Good for factual questions, definitions, and detailed descriptions.
# 2. kg_multi_hop_search: Search for entity relationships in Knowledge Graph. Good for multi-hop questions (e.g., family ties, founding relations, birth places).

# WORKFLOW:
# 1. **Analyze**: Understand the user's question. Identify key entities and the type of information needed.
# 2. **Retrieve**: 
#    - For general facts or descriptive answers, use `vector_search`.
#    - For relationship-based questions (who is related to whom, who founded what, etc.), use `kg_multi_hop_search`.
#    - **Important**: If one tool returns no relevant results or low-quality results, TRY THE OTHER TOOL. Or, rewrite the query to be more specific (e.g., extract just the entity name) and search again.
#    - You can use both tools if the question is complex.
# 3. **Evaluate**: Check if the retrieved information (Observation) answers the question.
#    - If the observation says "No relevant information found", you MUST try a different search strategy or query variation.
#    - If you have enough information, proceed to answer.
# 4. **Answer**: Provide a concise, accurate answer based ONLY on the retrieved context. 

# IMPORTANT RULES:
# - Think step-by-step in your "Thought" process.
# - Do NOT make up information. If the context doesn't contain the answer after multiple attempts, state that you cannot find it.
# - Keep your final answer direct and helpful.
# - You do NOT need to call a specific "finalize" tool. When you have the answer, just output it as your final response.

# Example Thought Process:
# Thought: I need to find who the mother of Lothair II is. This is a relationship question. I should try KG first.
# Action: kg_multi_hop_search
# Action Input: {"query": "Lothair II mother"}
# Observation: No relevant information found in knowledge graph.
# Thought: KG failed. Let me try vector search with a more specific query.
# Action: vector_search
# Action Input: {"query": "Lothair II mother name"}
# Observation: Source 1 (Similarity: 0.85): Ermengarde of Tours was the mother of Lothair II...
# Thought: I found the answer in the vector search results.
# Final Answer: Lothair II's mother was Ermengarde of Tours.
# """
        
#         # 使用 create_agent 创建智能体
#         agent = create_agent(
#             model=self.llm,
#             tools=self.tools,
#             system_prompt=system_prompt
#         )
        
#         print("[ReActRAGAgent] Agent created successfully with optimized prompt")
#         return agent
    
#     def run(self, question: str) -> Dict[str, Any]:
#         """
#         运行智能体回答问题
        
#         Args:
#             question: 用户问题
            
#         Returns:
#             包含答案和中间步骤的字典
#         """
#         print(f"\n{'='*80}")
#         print(f"[ReActRAGAgent] Processing question: {question}")
#         print(f"{'='*80}\n")
        
#         try:
#             # 构建消息列表
#             messages = [
#                 HumanMessage(content=question)
#             ]
            
#             # 调用智能体
#             result = self.agent.invoke({
#                 "messages": messages
#             })
            
#             # 提取答案
#             answer = self._extract_final_answer(result)
            
#             # 构建响应
#             response = {
#                 'question': question,
#                 'answer': answer,
#                 'raw_result': result,
#                 'success': True
#             }
            
#             # 保存对话历史
#             self.conversation_history.append({
#                 'question': question,
#                 'answer': answer
#             })
            
#             print(f"\n{'='*80}")
#             print(f"[ReActRAGAgent] Answer generated successfully")
#             print(f"{'='*80}\n")
            
#             return response
            
#         except Exception as e:
#             error_msg = f"Error during agent execution: {str(e)}"
#             print(f"[ReActRAGAgent] ERROR: {error_msg}")
#             import traceback
#             traceback.print_exc()
            
#             return {
#                 'question': question,
#                 'answer': f"Sorry, I encountered an error: {str(e)}",
#                 'success': False,
#                 'error': str(e)
#             }
    
#     def _extract_final_answer(self, result: Dict) -> str:
#         """从结果中提取最终答案"""
#         # 在 LangChain Agent 中，最后一条 AI 消息通常包含最终回答
#         if 'messages' in result:
#             messages = result['messages']
#             # 遍历找到最后一条 AI 消息
#             for msg in reversed(messages):
#                 if isinstance(msg, AIMessage):
#                     content = msg.content if hasattr(msg, 'content') else str(msg)
                    
#                     # 清理可能存在的 Thought/Action 残留标记（如果模型没有完美遵循停止符）
#                     # 通常 create_agent 会处理好，但为了鲁棒性，我们可以尝试提取 Final Answer 之后的内容
#                     # 或者如果整个消息就是答案，则直接返回
                    
#                     # 检查是否有明确的 Final Answer 标记
#                     if "Final Answer:" in content:
#                         return content.split("Final Answer:")[-1].strip()
                    
#                     # 如果没有标记，且消息不包含 Action/Thought 关键字，则视为最终答案
#                     if "Action:" not in content and "Thought:" not in content:
#                         return content.strip()
                        
#                     # 如果仍然包含结构化的 ReAct 步骤，取最后一部分作为尝试
#                     # 这种情况通常意味着模型没有正确停止，返回原始内容供调试
#                     return content.strip()
        
#         return str(result)
    
#     def get_conversation_summary(self) -> str:
#         """获取对话摘要"""
#         if not self.conversation_history:
#             return "No conversations yet."
        
#         summary = f"Total conversations: {len(self.conversation_history)}\n\n"
#         for i, conv in enumerate(self.conversation_history[-5:], 1):
#             summary += f"{i}. Q: {conv['question'][:80]}...\n"
#             summary += f"   A: {conv['answer'][:100]}...\n\n"
        
#         return summary
    
#     def reset(self):
#         """重置智能体状态"""
#         self.conversation_history = []
#         print("[ReActRAGAgent] Agent state reset")


# def create_rag_agent() -> ReActRAGAgent:
#     """工厂函数：创建 RAG 智能体实例"""
#     return ReActRAGAgent()


# if __name__ == "__main__":
#     # 测试智能体
#     agent = create_rag_agent()
    
#     # 测试问题
#     test_questions = [
#         "When did Lothair Ii's mother die?",
#         "Which film was released first, Aas Ka Panchhi or Phoolwari?",
#         "What is the place of birth of the performer of song Changed It?"
#     ]
    
#     for question in test_questions:
#         result = agent.run(question)
#         print(f"\nQuestion: {question}")
#         print(f"Answer: {result['answer']}")
#         print("-" * 80)