import sys
import os
# 强制将项目根目录加入Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import chromadb
from chromadb.config import Settings
from copy import deepcopy
from src.utils import compute_mdhash_id
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer


class ChromaEmbeddingStore:
    def __init__(self, embedding_model, persist_directory, batch_size, namespace):
        """
        初始化ChromaDB向量存储
        
        Args:
            embedding_model: SentenceTransformer模型实例
            persist_directory: ChromaDB持久化目录路径
            batch_size: 批处理大小
            namespace: 命名空间前缀
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.batch_size = batch_size
        self.namespace = namespace
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=namespace,
            metadata={"description": f"Embedding store for {namespace}"}
        )
        
        print(f"[{self.namespace}] Initialized ChromaDB collection at {persist_directory}")
    
    def _split_into_sentences(self, text):
        """
        将文本拆分为句子列表
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        if not text:
            return []
        
        # 简单的句子分割规则：基于标点符号 (. ! ?) 分割
        # 保留分隔符以便后续处理，或者去除空白
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 过滤空字符串和过短的句子
        filtered_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        
        return filtered_sentences

    def insert_text(self, text_list):
        """
        插入文本列表到向量数据库 (按句子级别拆分)
        
        Args:
            text_list: 需要插入的文本块列表 (chunks)
        """
        nodes_dict = {}
        
        # 遍历每个文本块，将其拆分为句子
        for block_idx, text in enumerate(text_list):
            sentences = self._split_into_sentences(text)
            
            for sent_idx, sentence in enumerate(sentences):
                # 对句子内容计算哈希ID，实现句子级别的去重
                hash_id = compute_mdhash_id(sentence, prefix=self.namespace + "-")
                
                # 如果该句子已存在字典中（可能在同一个block或不同block中出现），则跳过或覆盖
                # 这里选择保留第一个遇到的，或者可以根据需求添加元数据记录来源block_idx
                if hash_id not in nodes_dict:
                    nodes_dict[hash_id] = {
                        'content': sentence,
                        # 可选：存储来源信息，方便调试
                        # 'metadata': {'block_idx': block_idx, 'sent_idx': sent_idx} 
                    }
        
        all_hash_ids = list(nodes_dict.keys())
        
        # 检查已存在的ID
        existing_ids = self._get_existing_ids(all_hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing_ids]
        
        if not missing_ids:
            print(f"[{self.namespace}] All sentences already exist, skipping insertion")
            return
        
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        
        # 批量编码
        all_embeddings = self.embedding_model.encode(
            texts_to_encode,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size
        )
        
        # 插入到ChromaDB
        self._upsert(missing_ids, texts_to_encode, all_embeddings)
        print(f"[{self.namespace}] Inserted {len(missing_ids)} new sentence records")

    def _get_existing_ids(self, hash_ids):
        """
        获取已存在的ID列表
        
        Args:
            hash_ids: 需要检查的哈希ID列表
            
        Returns:
            已存在的ID集合
        """
        if not hash_ids:
            return set()
        
        try:
            # 尝试从ChromaDB中获取这些ID
            result = self.collection.get(ids=hash_ids)
            return set(result['ids'])
        except Exception as e:
            print(f"[{self.namespace}] Error checking existing IDs: {e}")
            return set()

    def _upsert(self, hash_ids, texts, embeddings):
        """
        更新或插入数据到ChromaDB
        
        Args:
            hash_ids: 哈希ID列表
            texts: 文本内容列表
            embeddings: 嵌入向量列表
        """
        # ChromaDB要求embeddings是列表格式
        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        
        # 分批插入以避免内存问题
        for i in range(0, len(hash_ids), self.batch_size):
            batch_end = min(i + self.batch_size, len(hash_ids))
            
            batch_ids = hash_ids[i:batch_end]
            batch_texts = texts[i:batch_end]
            batch_embeddings = embeddings_list[i:batch_end]
            
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings
            )


    def get_hash_id_to_text(self):
        """
        获取所有哈希ID到文本的映射
        
        Returns:
            字典，key为hash_id，value为text
        """
        # 获取集合中的所有数据
        result = self.collection.get()
        
        hash_id_to_text = {}
        
        ids = result.get('ids')
        documents = result.get('documents')
        
        if ids and documents:
            for hash_id, text in zip(ids, documents):
                if hash_id is not None and text is not None:
                    hash_id_to_text[hash_id] = text
        
        return deepcopy(hash_id_to_text)
    
    
    
    def encode_texts(self, texts):
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量数组
        """
        return self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size
        )
    
    def get_embeddings(self, hash_ids):
        """
        根据哈希ID获取嵌入向量
        
        Args:
            hash_ids: 哈希ID列表
            
        Returns:
            嵌入向量numpy数组
        """
        if not hash_ids:
            return np.array([])
        
        result = self.collection.get(ids=hash_ids, include=['embeddings'])
        
        embeddings = result.get('embeddings')
        if embeddings is None or len(embeddings) == 0:
            return np.array([])
        
        return np.array(embeddings)
    
    def search_similar(self, query_text, top_k=5):
        """
        搜索相似文本
        
        Args:
            query_text: 查询文本
            top_k: 返回最相似的K个结果
            
        Returns:
            包含ids、documents、distances的字典
        """
        # 编码查询文本
        query_embedding = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        
        # 执行相似度搜索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'distances']
        )
        
        return results
    
    def delete_by_ids(self, hash_ids):
        """
        根据ID删除记录
        
        Args:
            hash_ids: 需要删除的哈希ID列表
        """
        if hash_ids:
            self.collection.delete(ids=hash_ids)
            print(f"[{self.namespace}] Deleted {len(hash_ids)} records")
    
    def get_count(self):
        """
        获取集合中的记录总数
        
        Returns:
            记录数量
        """
        return self.collection.count()


if __name__ == '__main__':
    # 加载数据
    questions_path = "../dataset/2wikimultihop/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    chunks_path = "../dataset/2wikimultihop/chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # 注意：这里仍然传入chunks，但insert_text内部会将其拆分为sentences
    passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    print(f"First passage: {passages[0]}")
    print(f"Total chunks: {len(chunks)}")
    
    # 初始化embedding模型
    embedding_model = SentenceTransformer(
        "../models/Qwen3-Embedding-0.6B",
        device="cuda",
        local_files_only=True,
        trust_remote_code=False
    )
    
    # 初始化ChromaDB存储
    db_store = ChromaEmbeddingStore(
        embedding_model=embedding_model,
        persist_directory="../chroma_db",
        batch_size=32,
        namespace="2wikimultihop"
    )
    
    # 插入数据 (内部会自动拆分为句子)
    print("\nInserting sentences from chunks into ChromaDB...")
    db_store.insert_text(chunks)
    
    # 获取统计信息
    print(f"\nTotal sentence records in database: {db_store.get_count()}")
    
    # 测试相似度搜索
    if len(chunks) > 0:
        # 使用一个具体的句子或短查询进行测试
        query_text = "Who is the mother of Lothair II?"
        print(f"\nSearching for similar texts to: {query_text}...")
        results = db_store.search_similar(query_text, top_k=3)
        
        print("\nTop 3 similar results:")
        documents = results.get('documents')
        distances = results.get('distances')
        
        if documents and distances and len(documents) > 0 and len(distances) > 0:
            for i, (doc, distance) in enumerate(zip(documents[0], distances[0])):
                print(f"{i+1}. Distance: {distance:.4f}")
                print(f"   Text: {doc[:200]}...\n")
        else:
            print("No similar results found or results are empty.")
    
    
    # 测试获取embeddings
    hash_id_to_text = db_store.get_hash_id_to_text()
    if hash_id_to_text:
        sample_ids = list(hash_id_to_text.keys())[:3]
        print(f"\nRetrieving embeddings for {len(sample_ids)} sample IDs...")
        embeddings = db_store.get_embeddings(sample_ids)
        print(f"Embeddings shape: {embeddings.shape}")