"""
RAG Agent Tools - 为 ReAct 智能体提供的检索和评估工具
"""
import sys
import os
from typing import Optional, Dict, List, Any
from langchain_core.tools import tool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ROOT_DIR, dataset_name, embedding_model, llm_model, api_key, base_url, temperature, max_tokens
from src.embedding_store import ChromaEmbeddingStore
from src.kg_retriever import KnowledgeGraphRetriever
from src.utils import LLMModel
from sentence_transformers import SentenceTransformer


# 全局变量，用于缓存检索器实例（避免重复加载模型）
_vector_store = None
_kg_retriever = None
_embedding_model = None
_llm_model = None


def get_vector_store():
    """获取向量存储实例（单例模式）"""
    global _vector_store, _embedding_model

    if _vector_store is None:
        print("[AgentTools] Loading embedding model and vector store...")
        _embedding_model = SentenceTransformer(
            embedding_model,
            device="cuda",
            local_files_only=True
        )

        persist_directory = os.path.join(ROOT_DIR, "chroma_db")
        _vector_store = ChromaEmbeddingStore(
            embedding_model=_embedding_model,
            persist_directory=persist_directory,
            batch_size=32,
            namespace=dataset_name
        )
        print("[AgentTools] Vector store loaded successfully")

    return _vector_store


def get_kg_retriever():
    """获取知识图谱检索器实例（单例模式）"""
    global _kg_retriever

    if _kg_retriever is None:
        print("[AgentTools] Loading knowledge graph retriever...")
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")

        _kg_retriever = KnowledgeGraphRetriever(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            embedding_model_path=embedding_model
        )
        print("[AgentTools] Knowledge graph retriever loaded successfully")

    return _kg_retriever


@tool
def vector_search(query: str, top_k: int = 15) -> str:
    """
    Search for semantically similar documents in the vector database.
    Use this for factual questions, definitions, or detailed descriptions.
    Tip: If the initial search fails, try rewriting the query to be more specific or using synonyms.

    Args:
        query: The search query string. Keep it concise and keyword-rich.
        top_k: Number of results to return (default 5).

    Returns:
        A string containing the retrieved documents with their similarity scores.
    """
    try:
        vector_store = get_vector_store()
        # Increase top_k slightly to ensure recall, but filter by score if needed in future
        results = vector_store.search_similar(query, top_k=top_k)

        if not results or not results.get('documents') or not results['documents'][0]:
            return "No relevant documents found in vector database."

        formatted_results = []
        # Allow slightly longer context for better understanding
        max_doc_len = 500
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            similarity = 1 - distance
            # Only include results with reasonable similarity if possible, otherwise return all
            clean_doc = doc[:max_doc_len] + "..." if len(doc) > max_doc_len else doc
            formatted_results.append(
                f"Source {i+1} (Similarity: {similarity:.2f}): {clean_doc}"
            )

        output = "\n".join(formatted_results)
        return output

    except Exception as e:
        return f"Error during vector search: {str(e)}"


def decide_retrieval_strategy(query: str) -> str:
    """
    分析用户查询，决定是否需要检索以及采用何种检索策略。

    Args:
        query: 用户的查询问题

    Returns:
        检索策略建议JSON字符串
    """
    query_lower = query.lower().strip()
    
    # 更精确的疑问词检测
    wh_words = ['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose']
    auxiliary_verbs = ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would']
    
    # 判断是否为疑问句
    is_question = (
        any(query_lower.startswith(wh) for wh in wh_words) or
        any(query_lower.startswith(aux) for aux in auxiliary_verbs) or
        query.endswith('?')
    )
    
    needs_retrieval = is_question or len(query.split()) > 3

    # 多跳关系指示词（更全面）
    multi_hop_indicators = [
        'married', 'spouse', 'wife', 'husband', 'partner',
        'born', 'birth', 'place of birth', 'born in',
        'father', 'mother', 'parent', 'son', 'daughter', 'child', 'children',
        'director', 'actor', 'actress', 'performer', 'singer',
        'author', 'writer', 'founder', 'ceo', 'president',
        'company', 'organization', 'institution', 'university', 'school',
        'film', 'movie', 'book', 'song', 'album', 'music']

    complex_query_indicators = ['and', 'or', 'but', 'also', 'as well as']

    has_multi_entity = any(indicator in query_lower for indicator in multi_hop_indicators)
    has_complex_structure = (query.count(' ') > 8 or 
                           sum(1 for ind in complex_query_indicators if ind in query_lower) >= 1)

    multi_hop = has_multi_entity or has_complex_structure

    # 实体类型判断
    entity_keywords = ['person', 'people', 'individual', 
                      'place', 'city', 'town', 'country', 'location', 'region',
                      'company', 'organization', 'institution', 'university', 'school',
                      'film', 'movie', 'book', 'song', 'album', 'music']

    # 更智能的检索类型选择
    if has_multi_entity and any(kw in query_lower for kw in ['person', 'people', 'individual']):
        retrieval_type = "kg"
    elif multi_hop:
        retrieval_type = "both"
    elif any(kw in query_lower for kw in entity_keywords):
        retrieval_type = "kg"
    else:
        retrieval_type = "vector"

    if not needs_retrieval:
        estimated_steps = 0
    elif multi_hop:
        estimated_steps = 3
    else:
        estimated_steps = 1

    result = {
        "needs_retrieval": str(needs_retrieval).lower(),
        "retrieval_type": retrieval_type,
        "multi_hop": str(multi_hop).lower(),
        "estimated_steps": estimated_steps,
        "reasoning": f"Query analysis: '{query}' contains {len(query.split())} words. Is question: {is_question}. Multi-hop: {multi_hop}. Recommended strategy: {retrieval_type} retrieval."
    }

    return str(result)


def get_llm_model():
    """获取 LLM 模型实例（单例模式）"""
    global _llm_model

    if _llm_model is None:
        print("[AgentTools] Loading LLM model...")
        _llm_model = LLMModel()
        print("[AgentTools] LLM model loaded successfully")

    return _llm_model


@tool
def generate_answer_from_context(question: str, context: str) -> str:
    """
    基于检索到的上下文和问题，使用 LLM 生成答案。
    
    这是核心的答案生成工具，它会：
    1. 将问题和检索到的上下文拼接
    2. 调用 LLM 进行推理和回答
    3. 返回生成的答案
    
    Args:
        question: 用户的问题
        context: 检索到的上下文信息（可以来自向量搜索或知识图谱）
        
    Returns:
        LLM 生成的答案
    """
    try:
        llm = get_llm_model()
        
        # 构建更强大的系统提示
        system_prompt = """You are an expert question-answering assistant. Your task is to provide accurate, concise, and complete answers based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
1. READ CAREFULLY: Thoroughly analyze both the question and ALL provided context
2. BE PRECISE: Extract exact information from the context - do not paraphrase unnecessarily
3. BE COMPLETE: Include all relevant details needed to fully answer the question
4. NO HALLUCINATION: If the context doesn't contain sufficient information to answer confidently, respond with: "I cannot find sufficient information in the provided context to answer this question."
5. DIRECT ANSWER: Start your response directly with the answer, without introductory phrases like "Based on the context..." or "According to the information..."
6. CONCISE BUT COMPLETE: Be as brief as possible while including all necessary information

ANSWER FORMAT:
- For factual questions (who, what, when, where): Provide the specific fact directly
- For relationship questions: Clearly state the relationship
- For list questions: Provide items separated by commas
- Always maintain factual accuracy above all else

Example good answers:
- "Lothair II's mother was Ermengarde of Tours."
- "Janis Joplin was born in Port Arthur, Texas."
- "The film Aas Ka Panchhi was released in 1961."

Example bad answers:
- "Based on the context, Lothair II's mother appears to be Ermengarde of Tours." (unnecessary hedging)
- "I think Janis Joplin was born in Texas somewhere." (vague and uncertain)
"""
        
        # 构建用户提示
        user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        # 调用 LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        result = llm.infer(messages)
        
        # 清理答案
        answer = result.strip()
        
        # 如果答案包含不必要的前缀，移除它们
        prefixes_to_remove = [
            "Answer:", "Final Answer:", "The answer is:", "Based on the context:",
            "According to the provided information:", "From the context:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break
        
        return answer
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"


@tool
def evaluate_answer_quality(question: str, answer: str, context: str = "") -> str:
    """
    评估生成答案的质量，判断是否需要进一步检索。
    
    Args:
        question: 原始问题
        answer: 生成的答案
        context: 用于生成答案的上下文（可选）
        
    Returns:
        评估结果JSON字符串
    """
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    if not answer or len(answer.strip()) < 5:
        return str({
            "quality_score": 1,
            "is_complete": "false",
            "needs_more_retrieval": "true",
            "confidence": "low",
            "feedback": "Answer is too short or empty. Need more retrieval."
        })
    
    # 检查是否是无法回答的情况
    if "cannot find" in answer_lower or "insufficient" in answer_lower:
        return str({
            "quality_score": 2,
            "is_complete": "false",
            "needs_more_retrieval": "true",
            "confidence": "low",
            "feedback": "LLM indicates insufficient context. Need more retrieval."
        })
    
    uncertainty_words = ['maybe', 'perhaps', 'possibly', 'uncertain', 'unknown', '不确定', '可能', '也许', '不知道']
    has_uncertainty = any(word in answer_lower for word in uncertainty_words)
    
    answer_length_ok = len(answer.split()) >= 3
    
    question_starts = question_lower.split()[0] if question_lower.split() else ""
    direct_answer = False
    
    if question_starts in ['who', 'whom', 'whose', '谁']:
        capitalized_words = [w for w in answer.split() if w and len(w) > 1 and w[0].isupper()]
        direct_answer = len(capitalized_words) >= 1
    elif question_starts in ['what', 'which', '什么', '哪个']:
        direct_answer = len(answer.split()) >= 3
    elif question_starts in ['where', '哪里']:
        location_keywords = ['city', 'country', 'province', 'located', 'in', 'at']
        direct_answer = any(kw in answer_lower for kw in location_keywords)
    elif question_starts in ['when', '何时']:
        import re
        date_pattern = r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'
        direct_answer = bool(re.search(date_pattern, answer))
    else:
        direct_answer = answer_length_ok
    
    quality_score = 5
    if direct_answer:
        quality_score += 2
    if answer_length_ok:
        quality_score += 1
    if not has_uncertainty:
        quality_score += 1
    if context and len(context) > 100:
        quality_score += 1
    
    quality_score = min(quality_score, 10)
    
    needs_more_retrieval = (quality_score < 6 or has_uncertainty or not direct_answer)
    
    if quality_score >= 8:
        confidence = "high"
    elif quality_score >= 5:
        confidence = "medium"
    else:
        confidence = "low"
    
    feedback_parts = []
    if not direct_answer:
        feedback_parts.append("Answer doesn't directly address the question.")
    if has_uncertainty:
        feedback_parts.append("Answer contains uncertainty expressions.")
    if not answer_length_ok:
        feedback_parts.append("Answer is too brief, needs more details.")
    if quality_score < 6:
        feedback_parts.append("Overall quality is low, consider additional retrieval.")
    
    feedback = " ".join(feedback_parts) if feedback_parts else "Answer quality is acceptable."
    
    result = {
        "quality_score": quality_score,
        "is_complete": str(direct_answer and answer_length_ok).lower(),
        "needs_more_retrieval": str(needs_more_retrieval).lower(),
        "confidence": confidence,
        "feedback": feedback
    }
    
    return str(result)


@tool
def combine_retrieval_results(vector_results: str = "", kg_results: str = "") -> str:
    """
    合并来自向量检索和知识图谱检索的结果。
    
    Args:
        vector_results: 向量检索结果
        kg_results: 知识图谱检索结果
        
    Returns:
        合并后的综合结果字符串
    """
    combined_parts = []
    
    if vector_results and vector_results != "No relevant documents found in vector database.":
        combined_parts.append("## Vector Search Results:\n")
        combined_parts.append(vector_results)
    
    if kg_results and kg_results != "No relevant information found in knowledge graph.":
        combined_parts.append("\n## Knowledge Graph Results:\n")
        combined_parts.append(kg_results)
    
    if not combined_parts:
        return "No relevant information retrieved from any source."
    
    combined = "\n\n".join(combined_parts)
    
    summary_hint = (
        "\n\n## Integration Note:\n"
        "Please synthesize the information from both sources and provide a comprehensive answer."
    )
    
    return combined + summary_hint


@tool
def finalize_answer(question: str, retrieved_context: str, draft_answer: str) -> str:
    """
    验证并最终确定答案。
    
    Args:
        question: 原始问题
        retrieved_context: 检索到的上下文信息
        draft_answer: 草稿答案
        
    Returns:
        最终确认的答案或继续检索指示
    """
    context_length = len(retrieved_context) if retrieved_context else 0
    answer_length = len(draft_answer) if draft_answer else 0
    
    if context_length < 50:
        return (
            "INSUFFICIENT_CONTEXT: The retrieved context is too limited. "
            "Please perform additional retrieval to gather more evidence."
        )
    
    if answer_length < 3:
        return (
            "ANSWER_TOO_BRIEF: The draft answer is too brief. "
            "Please elaborate with more details from the retrieved context."
        )
    
    # 检查是否是错误消息
    if draft_answer.startswith("Error"):
        return (
            "ERROR_IN_ANSWER: There was an error generating the answer. "
            "Please try again or use a different approach."
        )
    
    # 如果可以接受，返回最终答案
    return f"FINAL_ANSWER_READY: {draft_answer}"

# """
# RAG Agent Tools - Optimized for Better Retrieval
# 专注于提供高质量的检索上下文，移除冗余的后处理工具
# """
# import sys
# import os
# from typing import Optional, Dict, List, Any
# from langchain_core.tools import tool

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.config import ROOT_DIR, dataset_name, embedding_model, llm_model, api_key, base_url, temperature, max_tokens
# from src.embedding_store import ChromaEmbeddingStore
# from src.kg_retriever import KnowledgeGraphRetriever
# from sentence_transformers import SentenceTransformer

# # 全局变量，用于缓存检索器实例（避免重复加载模型）
# _vector_store = None
# _kg_retriever = None
# _embedding_model = None


# def get_vector_store():
#     """获取向量存储实例（单例模式）"""
#     global _vector_store, _embedding_model
    
#     if _vector_store is None:
#         print("[AgentTools] Loading embedding model and vector store...")
#         _embedding_model = SentenceTransformer(
#             embedding_model,
#             device="cuda",
#             local_files_only=True
#         )
        
#         persist_directory = os.path.join(ROOT_DIR, "chroma_db")
#         _vector_store = ChromaEmbeddingStore(
#             embedding_model=_embedding_model,
#             persist_directory=persist_directory,
#             batch_size=32,
#             namespace=dataset_name
#         )
#         print("[AgentTools] Vector store loaded successfully")
    
#     return _vector_store


# def get_kg_retriever():
#     """获取知识图谱检索器实例（单例模式）"""
#     global _kg_retriever
    
#     if _kg_retriever is None:
#         print("[AgentTools] Loading knowledge graph retriever...")
#         neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
#         neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
#         neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")
        
#         _kg_retriever = KnowledgeGraphRetriever(
#             neo4j_uri=neo4j_uri,
#             neo4j_username=neo4j_username,
#             neo4j_password=neo4j_password,
#             embedding_model_path=embedding_model
#         )
#         print("[AgentTools] Knowledge graph retriever loaded successfully")
    
#     return _kg_retriever


# @tool
# def vector_search(query: str, top_k: int = 5) -> str:
#     """
#     Search for semantically similar documents in the vector database.
#     Use this for factual questions, definitions, or detailed descriptions.
#     Tip: If the initial search fails, try rewriting the query to be more specific or using synonyms.
    
#     Args:
#         query: The search query string. Keep it concise and keyword-rich.
#         top_k: Number of results to return (default 5).
        
#     Returns:
#         A string containing the retrieved documents with their similarity scores.
#     """
#     try:
#         vector_store = get_vector_store()
#         # Increase top_k slightly to ensure recall, but filter by score if needed in future
#         results = vector_store.search_similar(query, top_k=top_k)
        
#         if not results or not results.get('documents') or not results['documents'][0]:
#             return "No relevant documents found in vector database."
        
#         formatted_results = []
#         # Allow slightly longer context for better understanding
#         max_doc_len = 500
#         for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
#             similarity = 1 - distance
#             # Only include results with reasonable similarity if possible, otherwise return all
#             clean_doc = doc[:max_doc_len] + "..." if len(doc) > max_doc_len else doc
#             formatted_results.append(
#                 f"Source {i+1} (Similarity: {similarity:.2f}): {clean_doc}"
#             )
        
#         output = "\n".join(formatted_results)
#         return output
        
#     except Exception as e:
#         return f"Error during vector search: {str(e)}"


# @tool
# def kg_multi_hop_search(query: str, max_hops: int = 2) -> str:
#     """
#     Perform multi-hop retrieval in the Knowledge Graph to find complex relationships between entities.
#     Use this when the question involves relationships (e.g., "who is the father of...", "company founded by...", "born in").
    
#     Args:
#         query: The question or entity relationship to search for. Extract key entities if possible.
#         max_hops: Maximum number of hops in the graph (default 2).
        
#     Returns:
#         A string containing matched sentences and graph paths.
#     """
#     try:
#         kg_retriever = get_kg_retriever()
#         results = kg_retriever.retrieve_by_query(
#             query=query,
#             top_k_sentences=5, # Increase recall
#             max_hops=max_hops
#         )
        
#         output_parts = []
        
#         if results.get('matched_sentences'):
#             output_parts.append("### Direct Matches:")
#             for sent in results['matched_sentences'][:5]:
#                 output_parts.append(f"- {sent['content']}")
        
#         if results.get('paths'):
#             output_parts.append("\n### Relationship Paths:")
#             for path in results['paths'][:5]:
#                 path_str = " -> ".join(path['path'])
#                 output_parts.append(f"- Path: {path_str}")
        
#         if results.get('related_sentences'):
#             output_parts.append("\n### Related Context:")
#             seen_ids = set()
#             for sent in results['related_sentences'][:5]:
#                 if sent['sentence_id'] not in seen_ids:
#                     seen_ids.add(sent['sentence_id'])
#                     output_parts.append(f"- {sent['content']}")
        
#         if not output_parts:
#             return "No relevant information found in knowledge graph."
        
#         return "\n".join(output_parts)
        
#     except Exception as e:
#         return f"Error during knowledge graph search: {str(e)}"










