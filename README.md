# RAG检索API

一个基于FastAPI的后端API，实现了文档分割、向量数据库构建、BM25倒排索引构建以及混合检索功能，并集成了vLLM用于大模型推理。

## 功能特性

- **文档分割**：使用自定义的文本分割器将长文档分割为固定大小的块（无需LangChain）
- **向量检索**：基于Sentence Transformers和FAISS的高效向量相似度检索
- **BM25检索**：基于词频的传统倒排索引检索
- **混合检索**：结合向量检索和BM25检索的优势，提供更准确的结果
- **RAG生成**：集成vLLM大模型，基于检索结果生成自然语言响应
- **RESTful API**：提供简洁易用的API接口

## 技术栈

- FastAPI：Web框架
- Sentence Transformers：向量嵌入模型
- FAISS：向量数据库
- rank-bm25：BM25算法实现
- vLLM：高性能大模型推理引擎
- NLTK：自然语言处理工具

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 运行

```bash
# 启动API服务器
python main.py
```

服务器将在 `http://localhost:8000` 运行。

## API端点

### 1. 处理文档

上传文档并构建索引

```
POST /process_document/
```

**请求参数**：
- `file`：要上传的文本文件

**响应示例**：
```json
{
  "message": "文档处理完成",
  "num_chunks": 10,
  "dimension": 384
}
```

### 2. 向量检索

基于向量相似度检索文档块

```
POST /vector_search/
```

**请求参数**：
- `query`：查询文本
- `top_k`：返回结果数量（默认：5）

**响应示例**：
```json
{
  "results": [
    {
      "chunk_id": 0,
      "content": "文档内容...",
      "score": 0.98
    }
  ]
}
```

### 3. BM25检索

基于BM25算法检索文档块

```
POST /bm25_search/
```

**请求参数**：
- `query`：查询文本
- `top_k`：返回结果数量（默认：5）

**响应示例**：
```json
{
  "results": [
    {
      "chunk_id": 0,
      "content": "文档内容...",
      "score": 2.5
    }
  ]
}
```

### 4. 混合检索

结合向量检索和BM25检索

```
POST /hybrid_search/
```

**请求参数**：
- `query`：查询文本
- `top_k`：返回结果数量（默认：5）
- `alpha`：向量检索权重（0-1，默认：0.5）

**响应示例**：
```json
{
  "results": [
    {
      "chunk_id": 0,
      "content": "文档内容...",
      "vector_score": 0.98,
      "bm25_score": 2.5,
      "hybrid_score": 0.74
    }
  ]
}
```

### 5. RAG生成

结合检索结果和vLLM大模型生成响应

```
POST /rag_generate/
```

**请求参数**：
- `query`：查询文本
- `top_k`：检索结果数量（默认：5）
- `alpha`：向量检索权重（0-1，默认：0.5）
- `vllm_url`：远程vLLM API URL（默认：http://localhost:8000）

**响应示例**：
```json
{
  "retrieved_chunks": [
    {
      "chunk_id": 0,
      "content": "人工智能概述...",
      "vector_score": 0.98,
      "bm25_score": 2.5,
      "hybrid_score": 0.74
    }
  ],
  "generated_response": "人工智能是计算机科学的一个分支，旨在创造能够模拟人类智能的机器。",
  "vllm_response": {
    "text": ["人工智能是计算机科学的一个分支，旨在创造能够模拟人类智能的机器。"],
    "generated_texts": ["人工智能是计算机科学的一个分支，旨在创造能够模拟人类智能的机器。"]
  }
}
```

### 6. 系统状态

获取当前系统状态

```
GET /status/
```

**响应示例**：
```json
{
  "vector_index_exists": true,
  "bm25_index_exists": true,
  "num_chunks": 10,
  "model_name": "all-MiniLM-L6-v2"
}
```

## 使用示例

### 上传文档

```bash
curl -X POST -F "file=@document.txt" http://localhost:8000/process_document/
```

### 向量检索

```bash
curl -X POST "http://localhost:8000/vector_search/?query=人工智能&top_k=5"
```

### BM25检索

```bash
curl -X POST "http://localhost:8000/bm25_search/?query=人工智能&top_k=5"
```

### 混合检索

```bash
curl -X POST "http://localhost:8000/hybrid_search/?query=人工智能&top_k=5&alpha=0.7"
```

### RAG生成

```bash
curl -X POST "http://localhost:8000/rag_generate/?query=人工智能的定义&top_k=3"
```

## 注意事项

1. 首次运行RAG生成时会自动下载vLLM模型（默认：Qwen/Qwen-7B-Chat）
2. 目前仅支持文本文件的处理
3. 建议使用小于10MB的文档
4. 向量模型使用的是all-MiniLM-L6-v2，维度为384
5. vLLM模型加载需要一定时间和内存

## 性能优化

- 可以根据文档类型调整chunk_size和chunk_overlap参数
- 对于大型文档，可以考虑使用更高效的向量数据库（如Milvus、Pinecone）
- 可以更换更大的向量模型以提高检索准确性
- 可以调整vLLM的模型参数以平衡性能和质量

## 配置

可以通过修改main.py中的参数来配置系统：

- `chunk_size`：文档分割块大小（默认：512）
- `chunk_overlap`：文档分割块重叠大小（默认：50）
- `VLLM_MODEL_NAME`：vLLM模型名称（默认：Qwen/Qwen-7B-Chat）
- `VLLM_MAX_TOKENS`：vLLM生成最大 tokens 数（默认：512）
- `VLLM_TEMPERATURE`：vLLM生成温度（默认：0.7）