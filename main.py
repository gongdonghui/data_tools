# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
import os
import re
import nltk
import requests

# 下载nltk停用词
nltk.download('stopwords')
from nltk.corpus import stopwords

app = FastAPI(title="RAG检索API", description="向量检索与BM25检索的混合API")

# 初始化模型和全局变量
model = SentenceTransformer('all-MiniLM-L6-v2')

# 自定义文本分割函数
class CustomTextSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text):
        # 先按段落分割
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks = []
        
        for para in paragraphs:
            if not para:
                continue
            
            # 按句子分割段落
            sentences = re.split(r'(?<=[。！？；.!?;])\s*', para)
            current_chunk = []
            current_length = 0
            
            for sent in sentences:
                if not sent:
                    continue
                
                sent_length = len(sent)
                
                # 如果当前块加上新句子超过chunk_size
                if current_length + sent_length > self.chunk_size:
                    # 保存当前块
                    if current_chunk:
                        chunks.append(''.join(current_chunk))
                    
                    # 开始新块，保留重叠部分
                    overlap = ''.join(current_chunk[-self.chunk_overlap//len(current_chunk):]) if current_chunk else ''
                    current_chunk = [overlap, sent]
                    current_length = len(overlap) + sent_length
                else:
                    current_chunk.append(sent)
                    current_length += sent_length
            
            # 保存最后一个块
            if current_chunk:
                chunks.append(''.join(current_chunk))
        
        return chunks

text_splitter = CustomTextSplitter(chunk_size=512, chunk_overlap=50)

# 全局存储
vector_index = None
bm25_index = None
document_chunks = []
chunk_embeddings = []

@app.post("/process_document/")
async def process_document(file: UploadFile = File(...)):
    """上传文档并处理：分割、构建向量索引和BM25索引"""
    global vector_index, bm25_index, document_chunks, chunk_embeddings
    
    # 读取文件内容
    content = await file.read()
    text = content.decode('utf-8')
    
    # 文档分割（使用自定义分割器）
    document_chunks = text_splitter.split_text(text)
    if not document_chunks:
        raise HTTPException(status_code=400, detail="文档分割失败")
    
    # 生成向量嵌入
    chunk_embeddings = model.encode(document_chunks, convert_to_numpy=True)
    
    # 构建向量索引
    dimension = chunk_embeddings.shape[1]
    vector_index = faiss.IndexFlatIP(dimension)
    vector_index.add(chunk_embeddings)
    
    # 构建BM25索引
    tokenized_chunks = [chunk.split() for chunk in document_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    
    return JSONResponse(
        content={
            "message": "文档处理完成",
            "num_chunks": len(document_chunks),
            "dimension": dimension
        }
    )

@app.post("/vector_search/")
async def vector_search(query: str, top_k: int = 5):
    """向量检索"""
    if vector_index is None:
        raise HTTPException(status_code=400, detail="请先上传并处理文档")
    
    # 生成查询向量
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # 向量检索
    distances, indices = vector_index.search(query_embedding, top_k)
    
    # 整理结果
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(document_chunks):
            results.append({
                "chunk_id": int(idx),
                "content": document_chunks[idx],
                "score": float(distances[0][i])
            })
    
    return JSONResponse(content={"results": results})

@app.post("/bm25_search/")
async def bm25_search(query: str, top_k: int = 5):
    """BM25检索"""
    if bm25_index is None:
        raise HTTPException(status_code=400, detail="请先上传并处理文档")
    
    # BM25检索
    tokenized_query = query.split()
    scores = bm25_index.get_scores(tokenized_query)
    
    # 获取top_k结果
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # 整理结果
    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": int(idx),
            "content": document_chunks[idx],
            "score": float(scores[idx])
        })
    
    return JSONResponse(content={"results": results})

@app.post("/hybrid_search/")
async def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5):
    """混合检索：结合向量检索和BM25检索"""
    if vector_index is None or bm25_index is None:
        raise HTTPException(status_code=400, detail="请先上传并处理文档")
    
    # 向量检索
    query_embedding = model.encode([query], convert_to_numpy=True)
    vec_distances, vec_indices = vector_index.search(query_embedding, top_k * 2)
    
    # BM25检索
    tokenized_query = query.split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
    
    # 合并结果
    all_indices = set(vec_indices[0].tolist() + bm25_top_indices.tolist())
    
    # 归一化分数
    vec_min = np.min(vec_distances[0])
    vec_max = np.max(vec_distances[0])
    bm25_min = np.min(bm25_scores)
    bm25_max = np.max(bm25_scores)
    
    merged_results = []
    for idx in all_indices:
        if idx >= len(document_chunks):
            continue
            
        # 获取向量分数
        vec_score = 0.0
        if idx in vec_indices[0]:
            vec_idx = list(vec_indices[0]).index(idx)
            vec_score = (vec_distances[0][vec_idx] - vec_min) / (vec_max - vec_min) if vec_max != vec_min else 0.0
        
        # 获取BM25分数
        bm25_score = (bm25_scores[idx] - bm25_min) / (bm25_max - bm25_min) if bm25_max != bm25_min else 0.0
        
        # 混合分数
        hybrid_score = alpha * vec_score + (1 - alpha) * bm25_score
        
        merged_results.append({
            "chunk_id": int(idx),
            "content": document_chunks[idx],
            "vector_score": float(vec_score),
            "bm25_score": float(bm25_score),
            "hybrid_score": float(hybrid_score)
        })
    
    # 按混合分数排序
    merged_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    return JSONResponse(content={"results": merged_results[:top_k]})

@app.get("/status/")
async def get_status():
    """获取系统状态"""
    return JSONResponse(
        content={
            "vector_index_exists": vector_index is not None,
            "bm25_index_exists": bm25_index is not None,
            "num_chunks": len(document_chunks),
            "model_name": "all-MiniLM-L6-v2"
        }
    )

@app.post("/rag_generate/")
async def rag_generate(query: str, top_k: int = 5, alpha: float = 0.5, vllm_url: str = "http://localhost:8000", use_rerank: bool = True, rerank_model: str = "BAAI/bge-reranker-large"):
    """混合检索并调用远程vLLM生成响应"""
    if vector_index is None or bm25_index is None:
        raise HTTPException(status_code=400, detail="请先上传并处理文档")
    
    # 混合检索
    query_embedding = model.encode([query], convert_to_numpy=True)
    vec_distances, vec_indices = vector_index.search(query_embedding, top_k * 2)
    
    tokenized_query = query.split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
    
    all_indices = set(vec_indices[0].tolist() + bm25_top_indices.tolist())
    
    vec_min = np.min(vec_distances[0])
    vec_max = np.max(vec_distances[0])
    bm25_min = np.min(bm25_scores)
    bm25_max = np.max(bm25_scores)
    
    merged_results = []
    for idx in all_indices:
        if idx >= len(document_chunks):
            continue
            
        vec_score = 0.0
        if idx in vec_indices[0]:
            vec_idx = list(vec_indices[0]).index(idx)
            vec_score = (vec_distances[0][vec_idx] - vec_min) / (vec_max - vec_min) if vec_max != vec_min else 0.0
        
        bm25_score = (bm25_scores[idx] - bm25_min) / (bm25_max - bm25_min) if bm25_max != bm25_min else 0.0
        
        hybrid_score = alpha * vec_score + (1 - alpha) * bm25_score
        
        merged_results.append({
            "chunk_id": int(idx),
            "content": document_chunks[idx],
            "vector_score": float(vec_score),
            "bm25_score": float(bm25_score),
            "hybrid_score": float(hybrid_score)
        })
    
    merged_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    retrieved_chunks = merged_results[:top_k]
    
    # 使用rerank模型进行精细排序
    if use_rerank:
        try:
            # 准备rerank请求数据
            rerank_request = {
                "query": query,
                "documents": [chunk["content"] for chunk in retrieved_chunks],
                "model": rerank_model
            }
            
            # 调用vLLM的rerank API
            rerank_response = requests.post(
                f"{vllm_url}/rerank",
                json=rerank_request
            )
            rerank_response.raise_for_status()
            rerank_result = rerank_response.json()
            
            # 重新排序检索结果
            reranked_chunks = []
            for idx, result in enumerate(rerank_result["results"]):
                chunk = retrieved_chunks[result["index"]].copy()
                chunk["rerank_score"] = float(result["relevance_score"])
                reranked_chunks.append(chunk)
            
            # 按rerank分数排序
            reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            retrieved_chunks = reranked_chunks
        except requests.exceptions.RequestException as e:
            # 如果rerank失败，打印错误信息并继续使用原始检索结果
            print(f"Rerank API调用失败: {str(e)}")
            pass
    
    # 构建prompt
    context = "\n".join([chunk["content"] for chunk in retrieved_chunks])
    prompt = f"基于以下上下文回答问题：\n{context}\n\n问题：{query}\n回答："
    
    # 调用远程vLLM API
    try:
        response = requests.post(
            f"{vllm_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95
            }
        )
        response.raise_for_status()  # 检查HTTP错误
        vllm_result = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"调用vLLM API失败：{str(e)}")
    
    return JSONResponse(
        content={
            "retrieved_chunks": retrieved_chunks,
            "generated_response": vllm_result["text"][0] if "text" in vllm_result else vllm_result.get("generated_text", ""),
            "vllm_response": vllm_result
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
