import os
import sys
from typing import List, Dict, Optional, Tuple, Any
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import compute_mdhash_id


class Neo4jKnowledgeGraph:
    """Neo4j 知识图谱存储和管理类"""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "12345678",
        database: str = "neo4j",
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """
        初始化 Neo4j 连接
        
        Args:
            uri: Neo4j 数据库 URI
            username: 用户名
            password: 密码
            database: 数据库名称
            embedding_model: 嵌入模型（可选）
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.embedding_model = embedding_model
        
        # 创建驱动实例
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(username, password)
        )
        
        # 验证连接
        try:
            self.driver.verify_connectivity()
            print(f"[Neo4jKG] Successfully connected to {uri}")
        except Exception as e:
            print(f"[Neo4jKG] Failed to connect to Neo4j: {e}")
            raise
        
        # 初始化索引和约束
        self._create_indexes_and_constraints()
    
    
    def _create_indexes_and_constraints(self):
            """创建必要的索引和约束"""
            with self.driver.session(database=self.database) as session:
                # 为实体创建唯一约束
                session.run("""
                    CREATE CONSTRAINT entity_name_constraint IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.name IS UNIQUE
                """)
                
                # 为 Sentence 创建唯一约束
                session.run("""
                    CREATE CONSTRAINT sentence_id_constraint IF NOT EXISTS
                    FOR (s:Sentence) REQUIRE s.sentence_id IS UNIQUE
                """)
                
                # 为 Entity 的 name 属性创建索引
                session.run("""
                    CREATE INDEX entity_name_index IF NOT EXISTS
                    FOR (e:Entity) ON (e.name)
                """)
                
                # 为 Entity 的 type 属性创建索引
                session.run("""
                    CREATE INDEX entity_type_index IF NOT EXISTS
                    FOR (e:Entity) ON (e.type)
                """)
                
                # 为 Sentence 的 content 创建索引
                session.run("""
                    CREATE INDEX sentence_content_index IF NOT EXISTS
                    FOR (s:Sentence) ON (s.content)
                """)
                
                print("[Neo4jKG] Indexes and constraints created successfully")
    
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            print("[Neo4jKG] Connection closed")
    
    def insert_chunk(self, chunk_id: str, content: str, metadata: Optional[Dict] = None):
        """
        插入文本块
        
        Args:
            chunk_id: 块的唯一 ID
            content: 文本内容
            metadata: 元数据（可选）
        """
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.content = $content,
                    c.metadata = $metadata,
                    c.timestamp = timestamp()
            """, chunk_id=chunk_id, content=content, metadata=metadata or {})
    
    def insert_entity(self, name: str, entity_type: str = "Unknown", properties: Optional[Dict] = None):
        """
        插入实体节点
        
        Args:
            name: 实体名称
            entity_type: 实体类型（如 Person, Location, Organization 等）
            properties: 额外属性（可选）
        """
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $entity_type,
                    e.properties = $properties,
                    e.timestamp = timestamp()
            """, name=name, entity_type=entity_type, properties=properties or {})
    
    def insert_relationship(
        self, 
        source_entity: str, 
        target_entity: str, 
        relationship_type: str = "RELATED_TO",
        properties: Optional[Dict] = None
    ):
        """
        插入实体间关系
        
        Args:
            source_entity: 源实体名称
            target_entity: 目标实体名称
            relationship_type: 关系类型
            properties: 关系属性（可选）
        """
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (source:Entity {name: $source})
                MATCH (target:Entity {name: $target})
                MERGE (source)-[r:`$rel_type`]->(target)
                SET r.properties = $properties,
                    r.timestamp = timestamp()
            """.replace("$rel_type", relationship_type),
            source=source_entity,
            target=target_entity,
            properties=properties or {}
            )
    
    def link_chunk_to_entities(self, chunk_id: str, entity_names: List[str]):
        """
        将文本块与实体关联
        
        Args:
            chunk_id: 文本块 ID
            entity_names: 实体名称列表
        """
        with self.driver.session(database=self.database) as session:
            for entity_name in entity_names:
                session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MATCH (e:Entity {name: $entity_name})
                    MERGE (c)-[:MENTIONS]->(e)
                """, chunk_id=chunk_id, entity_name=entity_name)
    
    def batch_insert_chunks_with_entities(
        self,
        chunks_data: List[Dict],
        batch_size: int = 100
    ):
        """
        批量插入文本块和实体
        
        Args:
            chunks_data: 包含 chunk_id, content, entities 的字典列表
            batch_size: 批处理大小
        """
        total = len(chunks_data)
        for i in range(0, total, batch_size):
            batch = chunks_data[i:i + batch_size]
            
            with self.driver.session(database=self.database) as session:
                # 批量插入 chunks
                for item in batch:
                    session.run("""
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.content = $content,
                            c.timestamp = timestamp()
                    """, chunk_id=item['chunk_id'], content=item['content'])
                    
                    # 插入实体并建立关联
                    for entity in item.get('entities', []):
                        session.run("""
                            MERGE (e:Entity {name: $name})
                            SET e.type = $type,
                                e.timestamp = timestamp()
                        """, name=entity['name'], type=entity.get('type', 'Unknown'))
                        
                        session.run("""
                            MATCH (c:Chunk {chunk_id: $chunk_id})
                            MATCH (e:Entity {name: $name})
                            MERGE (c)-[:MENTIONS]->(e)
                        """, chunk_id=item['chunk_id'], name=entity['name'])
                
                print(f"[Neo4jKG] Inserted batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")
    
    def add_entity_embedding(self, entity_name: str, embedding: List[float]):
        """
        为实体添加向量嵌入
        
        Args:
            entity_name: 实体名称
            embedding: 向量嵌入
        """
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (e:Entity {name: $name})
                SET e.embedding = $embedding
            """, name=entity_name, embedding=embedding)
    
     
    def semantic_search_entities(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        基于向量相似度搜索实体
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            匹配的实体列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                WITH e, 
                     reduce(score = 0.0, i IN range(0, size(e.embedding)-1) | 
                         score + (e.embedding[i] * $emb[i])) AS similarity
                ORDER BY similarity DESC
                LIMIT $top_k
                RETURN e.name AS name, e.type AS type, similarity
            """, emb=query_embedding, top_k=top_k)
            
            return [dict(record) for record in result]

    def get_entity_neighbors(
        self,
        entity_name: str,
        max_depth: int = 2,
        max_neighbors: int = 50
    ) -> Dict:
        """
        获取实体的邻居节点（支持多跳）

        Args:
            entity_name: 实体名称
            max_depth: 最大跳跃深度
            max_neighbors: 最大邻居数量

        Returns:
            包含节点和关系的字典
        """
        with self.driver.session(database=self.database) as session:
            # 使用字符串格式化插入深度值（Neo4j不支持参数化路径长度）
            query = f"""
                MATCH path = (start:Entity {{name: $name}})-[*1..{max_depth}]-(neighbor:Entity)
                WHERE neighbor <> start
                  AND ALL(node IN nodes(path) WHERE node:Entity)
                WITH DISTINCT neighbor, 
                     relationships(path) AS rels,
                     length(path) AS hop_count
                LIMIT $max_neighbors
                RETURN neighbor.name AS name, 
                       neighbor.type AS type,
                       hop_count,
                       [rel IN rels | type(rel)] AS relationship_types
            """
            result = session.run(query, name=entity_name, max_neighbors=max_neighbors)

            neighbors = []
            for record in result:
                neighbors.append({
                    'name': record['name'],
                    'type': record['type'],
                    'hop_count': record['hop_count'],
                    'relationship_types': record['relationship_types']
                })

            return {
                'center_entity': entity_name,
                'neighbors': neighbors,
                'total_neighbors': len(neighbors)
            }
    
    def multi_hop_query(
        self,
        start_entities: List[str],
        relationship_patterns: Optional[List[str]] = None,
        max_depth: int = 3,
        max_results: int = 100
    ) -> List[Dict]:
        """
        执行多跳查询
        
        Args:
            start_entities: 起始实体列表
            relationship_patterns: 关系模式过滤（可选）
            max_depth: 最大跳跃深度
            max_results: 最大结果数量
            
        Returns:
            查询结果列表
        """
        with self.driver.session(database=self.database) as session:
            if relationship_patterns:
                # 构建带关系过滤的查询
                rel_pattern = "|".join([f"`{rel}`" for rel in relationship_patterns])
                query = f"""
                    MATCH path = (start:Entity)-[:{rel_pattern}*1..{max_depth}]-(target:Entity)
                    WHERE start.name IN $start_entities
                    WITH DISTINCT target, 
                         length(path) AS hops,
                         [node IN nodes(path) | node.name] AS path_nodes,
                         [rel IN relationships(path) | type(rel)] AS path_relations
                    LIMIT $max_results
                    RETURN target.name AS name,
                           target.type AS type,
                           hops,
                           path_nodes,
                           path_relations
                    ORDER BY hops ASC
                """
            else:
                query = f"""
                    MATCH path = (start:Entity)-[*1..{max_depth}]-(target:Entity)
                    WHERE start.name IN $start_entities AND target <> start
                    WITH DISTINCT target, 
                         length(path) AS hops,
                         [node IN nodes(path) | node.name] AS path_nodes,
                         [rel IN relationships(path) | type(rel)] AS path_relations
                    LIMIT $max_results
                    RETURN target.name AS name,
                           target.type AS type,
                           hops,
                           path_nodes,
                           path_relations
                    ORDER BY hops ASC
                """
            
            result = session.run(
                query,
                start_entities=start_entities,
                max_results=max_results
            )
            
            return [dict(record) for record in result]
    
    def search_by_entity_name(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict]:
        """
        按名称搜索实体（模糊匹配）
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            匹配的实体列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                RETURN e.name AS name, e.type AS type
                LIMIT $top_k
            """, query=query, top_k=top_k)
            
            return [dict(record) for record in result]
    
    def get_related_chunks(
        self, 
        entity_name: str,
        max_chunks: int = 10
    ) -> List[Dict]:
        """
        获取与实体相关的文本块
        
        Args:
            entity_name: 实体名称
            max_chunks: 最大返回块数量
            
        Returns:
            相关文本块列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Entity {name: $name})<-[:MENTIONS]-(c:Chunk)
                RETURN c.chunk_id AS chunk_id, c.content AS content
                LIMIT $max_chunks
            """, name=entity_name, max_chunks=max_chunks)
            
            return [dict(record) for record in result]
    

    def semantic_search_sentences(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        基于向量相似度搜索句子
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            匹配的句子列表，包含句子内容、ID和相关实体
        """
        with self.driver.session(database=self.database) as session:
            # 修复：将参数名 query 改为 emb，避免与 session.run 的第一个位置参数冲突
            result = session.run("""
                MATCH (s:Sentence)
                WHERE s.embedding IS NOT NULL
                WITH s, 
                     reduce(score = 0.0, i IN range(0, size(s.embedding)-1) | 
                         score + (s.embedding[i] * $emb[i])) AS similarity
                ORDER BY similarity DESC
                LIMIT $top_k
                OPTIONAL MATCH (s)-[:MENTIONS]->(e:Entity)
                RETURN s.sentence_id AS sentence_id, 
                       s.content AS content, 
                       s.chunk_id AS chunk_id,
                       similarity,
                       collect(e.name) AS mentioned_entities
            """, emb=query_embedding, top_k=top_k)
            
            return [dict(record) for record in result]
        

    def get_graph_statistics(self) -> Dict:
        """获取图谱统计信息"""
        with self.driver.session(database=self.database) as session:
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()['count']
            
            # 统计 Sentence 数量
            sentence_count = session.run("MATCH (s:Sentence) RETURN count(s) AS count").single()['count']
            
            relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()['count']
            
            type_distribution = session.run("""
                MATCH (e:Entity)
                RETURN e.type AS type, count(e) AS count
                ORDER BY count DESC
                LIMIT 20
            """)
            
            type_dist_dict = {record['type']: record['count'] for record in type_distribution}
            
            return {
                'total_entities': entity_count,
                'total_sentences': sentence_count,
                'total_relationships': relationship_count,
                'entity_type_distribution': type_dist_dict
            }

    
    def clear_database(self):
        """清空数据库（谨慎使用）"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("[Neo4jKG] Database cleared")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()