# NEW_FILE_CODE
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ROOT_DIR, dataset_name
from src.kg_builder import KnowledgeGraphBuilder
from src.kg_retriever import KnowledgeGraphRetriever
from src.utils import LLMModel


def main():
    """主函数 - 演示知识图谱构建和检索"""
    llm_model = LLMModel()
    # 配置参数
    NEO4J_URI = os.getenv("NEO4J_URI", " bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
    
    EMBEDDING_MODEL_PATH = os.path.join(ROOT_DIR, "models", "Qwen3-Embedding-0.6B")
    CHUNKS_FILE = os.path.join(ROOT_DIR, "dataset", dataset_name, "chunks.json")
    
    print("=" * 80)
    print("Knowledge Graph Builder for RAG Multi-hop Retrieval")
    print("=" * 80)

    # Step 1: 构建知识图谱
    print("\n[Step 1] Building Knowledge Graph...")
    print("-" * 80)

    with KnowledgeGraphBuilder(
        neo4j_uri=NEO4J_URI,
        neo4j_username=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
        use_nlp=True,  # 如果没有安装 spaCy，设为 False
        embedding_model_path=EMBEDDING_MODEL_PATH
    ) as builder:
        builder.build_from_chunks_file(CHUNKS_FILE, batch_size=50)
    
    print("\n[Step 1] Knowledge Graph construction completed!\n")
    
    # Step 2: 测试检索
    print("[Step 2] Testing Knowledge Graph Retrieval...")
    print("-" * 80)
    
    with KnowledgeGraphRetriever(
        neo4j_uri=NEO4J_URI,
        neo4j_username=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
        embedding_model_path=EMBEDDING_MODEL_PATH
    ) as retriever:
        
        # 测试查询 1: 基于问题的检索
        test_queries = [
            "When did Lothair Ii's mother die?",
            "Which film was released first, Aas Ka Panchhi or Phoolwari?",
            "What is the place of birth of the performer of song Changed It?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("~" * 80)
            
            results = retriever.retrieve_by_query(
                query=query,
                top_k_entities=3,
                max_hops=2,
                max_chunks_per_entity=2
            )
            
            # 格式化输出
            formatted_context = retriever.format_for_rag(results)
            print(formatted_context)
            print()
            system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. 
                    Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. 
                    Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations."""
            user_prompt = f"Content: {formatted_context}\n Question: {query}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            print(f"messages: {messages}")
            result = llm_model.infer(messages)
            print(result)

        
        # 测试查询 2: 基于实体的检索
        print("\n" + "=" * 80)
        print("Testing Entity-based Retrieval")
        print("=" * 80)
        
        test_entities = ["Lothair Ii", "Aas Ka Panchhi", "Changed It"]
        
        for entity_name in test_entities:
            print(f"\nEntity: {entity_name}")
            print("~" * 80)
            
            results = retriever.retrieve_by_entity(
                entity_name=entity_name,
                max_hops=2,
                max_neighbors=10,
                max_chunks=5
            )
            
            print(f"Center Entity: {results['entity']}")
            print(f"Total Neighbors: {results['neighbors']['total_neighbors']}")
            
            if results['neighbors']['neighbors']:
                print("\nSample Neighbors:")
                for neighbor in results['neighbors']['neighbors'][:5]:
                    print(f"  - {neighbor['name']} ({neighbor['type']}) "
                          f"[{neighbor['hop_count']} hops via {neighbor['relationship_types']}]")
            
            if results['chunks']:
                print(f"\nRelated Chunks: {len(results['chunks'])}")
                for chunk in results['chunks'][:2]:
                    print(f"  - {chunk['chunk_id']}: {chunk['content'][:150]}...")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)



if __name__ == "__main__":
    main()