import json
import re
import sys
import os
from typing import List, Dict, Tuple, Set
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Using simple sentence splitting.")

from src.neo4j_store import Neo4jKnowledgeGraph
from sentence_transformers import SentenceTransformer


class KnowledgeGraphBuilder:
    """知识图谱构建器 - 从文本块中提取实体、关系和句子"""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password",
        use_nlp: bool = True,
        embedding_model_path: str = None
    ):
        self.use_nlp = use_nlp and SPACY_AVAILABLE
        self.nlp = None
        
        if self.use_nlp:
            try:
                # 使用 en_core_web_trf 进行句子切分和实体识别
                self.nlp = spacy.load("en_core_web_trf")
                print("[KGBuild] Loaded spaCy NLP model")
            except OSError:
                print("[KGBuild] spaCy model not found. Install with: python -m spacy download en_core_web_trf")
                self.use_nlp = False
        
        self.kg_store = Neo4jKnowledgeGraph(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        if embedding_model_path:
            self.embedding_model = SentenceTransformer(
                embedding_model_path,
                device="cuda",
                local_files_only=True
            )
            print(f"[KGBuild] Loaded embedding model from {embedding_model_path}")
        else:
            self.embedding_model = None
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        将文本切分为句子列表
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        if self.use_nlp and self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # 简单的正则切分
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def extract_entities_from_sentence(self, sentence: str) -> List[Dict]:
        """
        从单个句子中提取实体
        
        Args:
            sentence: 输入句子
            
        Returns:
            实体列表
        """
        if self.use_nlp and self.nlp:
            doc = self.nlp(sentence)
            entities = []
            seen = set()
            for ent in doc.ents:
                if ent.text not in seen and len(ent.text) > 2:
                    type_mapping = {
                        'PERSON': 'Person', 'GPE': 'Location', 'LOC': 'Location',
                        'ORG': 'Organization', 'NORP': 'Group', 'FAC': 'Facility',
                        'EVENT': 'Event', 'WORK_OF_ART': 'CreativeWork', 'PRODUCT': 'Product'
                    }
                    entities.append({
                        'name': ent.text,
                        'type': type_mapping.get(ent.label_, 'Unknown')
                    })
                    seen.add(ent.text)
            return entities
        else:
            # 简单规则提取（略，沿用之前的简单提取逻辑）
            return []
    
    def build_from_chunks_file(self, chunks_file: str, batch_size: int = 50):
        """
        从 chunks.json 文件构建句子级知识图谱
        
        Args:
            chunks_file: chunks JSON 文件路径
            batch_size: 批处理大小
        """
        print(f"[KGBuild] Loading chunks from {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"[KGBuild] Loaded {len(chunks)} chunks")
        
        all_sentences_data = []
        all_entities = set()
        sentence_count = 0
        
        for chunk_idx, chunk_text in enumerate(chunks):
            # 1. 切分句子
            sentences = self.split_into_sentences(chunk_text)
            
            for sent_idx, sentence in enumerate(sentences):
                if not sentence:
                    continue
                
                sentence_id = f"chunk_{chunk_idx}_sent_{sent_idx}"
                sentence_count += 1
                
                # 2. 提取句子中的实体
                entities = self.extract_entities_from_sentence(sentence)
                
                # 去重实体
                unique_entities = []
                seen_entities = set()
                for entity in entities:
                    if entity['name'] not in seen_entities:
                        seen_entities.add(entity['name'])
                        unique_entities.append(entity)
                        all_entities.add(entity['name'])
                
                all_sentences_data.append({
                    'sentence_id': sentence_id,
                    'content': sentence,
                    'chunk_id': f"chunk_{chunk_idx}",
                    'entities': unique_entities
                })
            
            if (chunk_idx + 1) % 100 == 0:
                print(f"[KGBuild] Processed {chunk_idx + 1}/{len(chunks)} chunks ({sentence_count} sentences)")
        
        print(f"[KGBuild] Extracted {sentence_count} sentences and {len(all_entities)} unique entities")
        
        # 3. 批量插入到 Neo4j
        print(f"[KGBuild] Inserting data into Neo4j...")
        self._batch_insert_sentences(all_sentences_data, batch_size)
        
        # 4. 为实体和句子添加嵌入向量
        if self.embedding_model:
            print(f"[KGBuild] Generating embeddings for entities and sentences...")
            self._generate_and_store_embeddings(all_sentences_data, all_entities, batch_size)
        
        print(f"[KGBuild] Sentence-level Knowledge Graph construction completed!")
        
        stats = self.kg_store.get_graph_statistics()
        print(f"\n[KGBuild] Graph Statistics:")
        print(f"  Total Entities: {stats['total_entities']}")
        print(f"  Total Sentences: {stats['total_sentences']}")
        print(f"  Total Relationships: {stats['total_relationships']}")
    
    def _batch_insert_sentences(self, sentences_data: List[Dict], batch_size: int):
        """批量插入句子和实体关系"""
        total = len(sentences_data)
        for i in range(0, total, batch_size):
            batch = sentences_data[i:i + batch_size]
            
            with self.kg_store.driver.session(database=self.kg_store.database) as session:
                for item in batch:
                    # 插入句子节点
                    session.run("""
                        MERGE (s:Sentence {sentence_id: $sentence_id})
                        SET s.content = $content,
                            s.chunk_id = $chunk_id,
                            s.timestamp = timestamp()
                    """, sentence_id=item['sentence_id'], content=item['content'], chunk_id=item['chunk_id'])
                    
                    # 插入实体并建立 MENTIONS 关系
                    for entity in item.get('entities', []):
                        session.run("""
                            MERGE (e:Entity {name: $name})
                            SET e.type = $type,
                                e.timestamp = timestamp()
                        """, name=entity['name'], type=entity.get('type', 'Unknown'))
                        
                        session.run("""
                            MATCH (s:Sentence {sentence_id: $sentence_id})
                            MATCH (e:Entity {name: $name})
                            MERGE (s)-[:MENTIONS]->(e)
                        """, sentence_id=item['sentence_id'], name=entity['name'])
                
                print(f"[Neo4jKG] Inserted sentence batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")
    
    def _generate_and_store_embeddings(self, sentences_data: List[Dict], all_entities: Set[str], batch_size: int):
        """生成并存储实体和句子的嵌入向量"""
        
        # 1. 实体嵌入
        entity_list = list(all_entities)
        for i in range(0, len(entity_list), batch_size):
            batch_entities = entity_list[i:i + batch_size]
            embeddings = self.embedding_model.encode(
                batch_entities,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            with self.kg_store.driver.session(database=self.kg_store.database) as session:
                for entity_name, embedding in zip(batch_entities, embeddings):
                    session.run("""
                        MATCH (e:Entity {name: $name})
                        SET e.embedding = $embedding
                    """, name=entity_name, embedding=embedding.tolist())
            
            if (i + batch_size) % 500 == 0:
                print(f"[KGBuild] Entity embeddings: {min(i + batch_size, len(entity_list))}/{len(entity_list)}")
        
        # 2. 句子嵌入
        sentence_texts = [item['content'] for item in sentences_data]
        sentence_ids = [item['sentence_id'] for item in sentences_data]
        
        for i in range(0, len(sentence_texts), batch_size):
            batch_texts = sentence_texts[i:i + batch_size]
            batch_ids = sentence_ids[i:i + batch_size]
            
            embeddings = self.embedding_model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            with self.kg_store.driver.session(database=self.kg_store.database) as session:
                for sent_id, embedding in zip(batch_ids, embeddings):
                    session.run("""
                        MATCH (s:Sentence {sentence_id: $sentence_id})
                        SET s.embedding = $embedding
                    """, sentence_id=sent_id, embedding=embedding.tolist())
            
            if (i + batch_size) % 500 == 0:
                print(f"[KGBuild] Sentence embeddings: {min(i + batch_size, len(sentence_texts))}/{len(sentence_texts)}")

    def close(self):
        self.kg_store.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()