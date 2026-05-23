# NEW_FILE_CODE
import sys
import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.neo4j_store import Neo4jKnowledgeGraph


class KnowledgeGraphRetriever:
    """知识图谱检索器 - 支持多跳查询和 RAG"""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password",
        embedding_model_path: str = None
    ):
        """
        初始化检索器
        
        Args:
            neo4j_uri: Neo4j 连接 URI
            neo4j_username: Neo4j 用户名
            neo4j_password: Neo4j 密码
            embedding_model_path: 嵌入模型路径
        """
        self.kg_store = Neo4jKnowledgeGraph(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # 加载嵌入模型
        if embedding_model_path:
            self.embedding_model = SentenceTransformer(
                embedding_model_path,
                device="cuda",
                local_files_only=True
            )
            print(f"[KGRetriever] Loaded embedding model from {embedding_model_path}")
        else:
            self.embedding_model = None
    
    def retrieve_by_query(
        self,
        query: str,
        top_k_sentences: int = 5,
        max_hops: int = 2,
        max_entities_per_sentence: int = 3
    ) -> Dict:
        """
        基于查询进行句子级多跳检索
        
        Args:
            query: 用户查询
            top_k_sentences: 初始匹配的句子数量
            max_hops: 最大跳跃步数
            max_entities_per_sentence: 每个句子关联的最大实体数
            
        Returns:
            检索结果，包含句子、实体和路径
        """
        results = {
            'query': query,
            'matched_sentences': [],
            'multi_hop_entities': [],
            'related_sentences': [],
            'paths': []
        }
        
        # Step 1: 向量搜索匹配的句子
        if self.embedding_model:
            query_embedding = self.embedding_model.encode(
                [query],
                normalize_embeddings=True
            )[0].tolist()
            
            matched_sentences = self.kg_store.semantic_search_sentences(
                query_embedding,
                top_k=top_k_sentences
            )
        else:
            matched_sentences = []
        
        results['matched_sentences'] = matched_sentences
        
        if not matched_sentences:
            print(f"[KGRetriever] No sentences matched for query: {query}")
            return results
        
        # Step 2: 从匹配的句子中提取实体，进行多跳查询
        all_mentioned_entities = []
        for sent in matched_sentences:
            all_mentioned_entities.extend(sent.get('mentioned_entities', []))
        
        # 去重
        start_entity_names = list(set(all_mentioned_entities))[:10]  # 限制起始实体数量
        
        if start_entity_names:
            multi_hop_results = self.kg_store.multi_hop_query(
                start_entities=start_entity_names,
                max_depth=max_hops,
                max_results=50
            )
            results['multi_hop_entities'] = multi_hop_results
            
            # Step 3: 获取多跳实体相关的句子
            all_entity_names = start_entity_names.copy()
            for result in multi_hop_results:
                if result['name'] not in all_entity_names:
                    all_entity_names.append(result['name'])
            
            for entity_name in all_entity_names[:10]:
                # 获取包含该实体的其他句子
                related_sents = self._get_sentences_by_entity(entity_name, max_chunks=3)
                results['related_sentences'].extend(related_sents)
        
        # 修复：添加 return 语句，返回检索结果字典
        return results

    def _get_sentences_by_entity(self, entity_name: str, max_chunks: int = 5) -> List[Dict]:
        """获取包含特定实体的句子"""
        try:
            with self.kg_store.driver.session(database=self.kg_store.database) as session:
                result = session.run("""
                    MATCH (e:Entity {name: $name})<-[:MENTIONS]-(s:Sentence)
                    RETURN s.sentence_id AS sentence_id, s.content AS content, s.chunk_id AS chunk_id
                    LIMIT $max_chunks
                """, name=entity_name, max_chunks=max_chunks)
                
                # 修复：显式调用 .data() 确保在 session 关闭前完全提取数据
                records = result.data()
                return records
        except Exception as e:
            print(f"[KGRetriever] Error fetching sentences for entity {entity_name}: {e}")
            return []

    def format_for_rag(self, retrieval_results: Dict) -> str:
        """
        将检索结果格式化为适合 LLM 的上下文（句子级）
        """
        context_parts = []
        
        # 添加匹配的句子
        if retrieval_results.get('matched_sentences'):
            context_parts.append("### Relevant Sentences (Vector Search):")
            for sent in retrieval_results['matched_sentences']:
                context_parts.append(
                    f"- [Similarity: {sent['similarity']:.4f}] {sent['content']}"
                )
        
        # 添加多跳路径信息
        if retrieval_results.get('paths'):
            context_parts.append("\n### Knowledge Graph Paths:")
            for path in retrieval_results['paths'][:5]:
                path_str = " -> ".join(path['path'])
                relations_str = " -> ".join(path['relations'])
                context_parts.append(
                    f"- Path: {path_str} ({path['hops']} hops)"
                )
        
        # 添加相关句子
        if retrieval_results.get('related_sentences'):
            context_parts.append("\n### Related Sentences (Graph Traversal):")
            seen_ids = set()
            for sent in retrieval_results['related_sentences'][:10]:
                if sent['sentence_id'] not in seen_ids:
                    seen_ids.add(sent['sentence_id'])
                    context_parts.append(f"- {sent['content']}")
        
        return "\n".join(context_parts)
    
    
    def retrieve_by_entity(
        self,
        entity_name: str,
        max_hops: int = 2,
        max_neighbors: int = 20,
        max_chunks: int = 10
    ) -> Dict:
        """
        基于实体名称进行检索
        
        Args:
            entity_name: 实体名称
            max_hops: 最大跳跃步数
            max_neighbors: 最大邻居数量
            max_chunks: 最大文本块数量
            
        Returns:
            检索结果
        """
        results = {
            'entity': entity_name,
            'neighbors': [],
            'chunks': []
        }
        
        # 获取邻居节点
        neighbors = self.kg_store.get_entity_neighbors(
            entity_name,
            max_depth=max_hops,
            max_neighbors=max_neighbors
        )
        
        results['neighbors'] = neighbors
        
        # 获取相关文本块
        chunks = self.kg_store.get_related_chunks(
            entity_name,
            max_chunks=max_chunks
        )
        
        results['chunks'] = chunks
        
        return results
    
    
    
    def close(self):
        """关闭资源"""
        self.kg_store.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()