# LA_RAG: 知识图谱增强的RAG智能体问答系统

基于知识图谱和ReAct智能体的多跳检索问答系统，支持复杂问题的推理和准确回答。

## 项目简介

LA_RAG (Knowledge Graph Enhanced RAG Agent System) 是一个完整的问答系统，整合了：
- **知识图谱构建**: 从文本块中自动提取实体、关系和语义信息
- **多跳检索**: 基于知识图谱的复杂推理检索
- **ReAct智能体**: 使用LangChain框架的推理-行动智能体
- **答案评估**: 支持LLM评估和包含匹配评估

## 主要特性

- 🧠 **智能知识图谱**: 自动构建和检索知识图谱
- 🔍 **多跳推理**: 支持复杂问题的多跳推理检索
- 🤖 **ReAct智能体**: 基于LangChain的智能问答代理
- 📊 **多种评估**: LLM评估和包含匹配评估
- ⚡ **批量处理**: 支持批量问答和中间结果保存
- 🔧 **灵活配置**: YAML配置文件和环境变量支持

## 技术栈

- **LLM**: Qwen3.5-plus (阿里云DashScope API)
- **Embedding**: Qwen3-Embedding-0.6B
- **知识图谱**: Neo4j
- **智能体框架**: LangChain
- **NLP处理**: spaCy
- **向量检索**: ChromaDB



## 快速开始

### 1. 配置系统

编辑 `config/config.yaml` 文件，设置相关参数：

```yaml
llm:
  model: "qwen3.5-plus"
  api_key: "your_api_key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  temperature: 0.0
  max_tokens: 16384

embedding:
  model: "path/to/Qwen3-Embedding-0.6B"
  device: "cuda:0"
  batch_size: 16
```

### 2. 构建知识图谱

```bash
python src/kg_main.py
```

### 3. 运行完整系统

```bash
python src/main.py
```

### 4. 测试智能体

```bash
python agent/agent_main.py
```

## 使用示例

### 完整系统使用

```python
from src.main import RAGAgentSystem

# 初始化系统
system = RAGAgentSystem(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    build_kg=False  # 如果知识图谱已构建
)

# 回答单个问题
question_data = {
    'id': '1',
    'question': 'When did Lothair Ii\'s mother die?',
    'answer': 'March 10, 965'
}

result = system.answer_question_with_gold(question_data)
print(f"Predicted: {result['pred_answer']}")

# 批量回答问题
results = system.answer_all_questions(
    questions_file='dataset/2wikimultihop/questions.json',
    output_file='results/predictions.json',
    max_questions=20
)

# 评估结果
llm_accuracy, contain_accuracy = system.evaluate_predictions(
    predictions_file='results/predictions.json'
)
```

### 知识图谱检索

```python
from src.kg_retriever import KnowledgeGraphRetriever

retriever = KnowledgeGraphRetriever(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    embedding_model_path="models/Qwen3-Embedding-0.6B"
)

# 基于问题检索
results = retriever.retrieve_by_query(
    query="When did Lothair Ii's mother die?",
    top_k_sentences=5,
    max_hops=2
)

# 基于实体检索
results = retriever.retrieve_by_entity(
    entity_name="Lothair Ii",
    max_hops=2,
    max_neighbors=10
)
```

### ReAct智能体使用

```python
from agent.react_agent import create_rag_agent

agent = create_rag_agent()

# 运行智能体
result = agent.run("When did Lothair Ii's mother die?")
print(f"Answer: {result['answer']}")
print(f"Success: {result['success']}")
```

## 数据集

项目使用公开数据集，这是一个多跳问答数据集，包含：
- `chunks.json`: 文本块数据
- `questions.json`: 问题和标准答案

## 评估方法

系统支持两种评估方式：

1. **LLM评估**: 使用大模型判断答案是否正确
2. **Contain评估**: 检查预测答案是否包含标准答案的关键信息

```python
from src.evaluate import evaluate_predictions
from src.utils import LLMModel

llm_model = LLMModel()
llm_accuracy, contain_accuracy = evaluate_predictions(
    predictions_path='results/predictions.json',
    llm_model=llm_model,
    max_workers=10
)
```

## 配置说明

### LLM配置
- `model`: 使用的模型名称
- `api_key`: API密钥
- `base_url`: API基础URL
- `temperature`: 温度参数
- `max_tokens`: 最大token数

### Embedding配置
- `model`: 嵌入模型路径
- `device`: 运行设备 (cuda/cpu)
- `batch_size`: 批处理大小

### Agent配置
- `max_loops`: 最大循环次数
- `max_token_budget`: 最大token预算
- `verbose`: 是否显示详细信息

## 常见问题

### 1. Neo4j连接失败
检查Neo4j服务是否启动，以及连接配置是否正确。

### 2. spaCy模型加载失败
确保已安装spaCy模型：`python -m spacy download en_core_web_trf`

### 3. GPU内存不足
将配置中的 `device` 改为 `"cpu"` 或减小 `batch_size`。

### 4. API调用失败
检查API密钥是否正确，以及网络连接是否正常。

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或联系项目维护者。

## 致谢

- Qwen团队提供的优秀模型
- LangChain社区的支持
- Neo4j图数据库
