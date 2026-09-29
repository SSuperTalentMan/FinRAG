#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_qa/core/document_processor.py — 文档加载 + 双层切分

支持的文件类型：
  .txt  → TextLoader
  .pdf  → OCRPDFLoader
  .docx → OCRDOCLoader
  .ppt/.pptx → OCRPPTLoader（预留接口）
  .jpg/.png → OCRIMGLoader（预留接口）
  .md   → UnstructuredMarkdownLoader

双层切分策略：
  1. 父块（Parent Chunk）：较大片段，保留完整语义单元
  2. 子块（Child Chunk）：较小片段，用于向量检索
  3. 子块元数据记录 parent_id 和 parent_content，便于检索后返回完整上下文
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter
from loguru import logger

from config import get_config
from ..edu_text_spliter import ChineseRecursiveTextSplitter
from ..edu_document_loaders import OCRPDFLoader, OCRDOCLoader

conf = get_config()

# 文件扩展名 → 加载器映射
DOCUMENT_LOADERS = {
    ".txt":  lambda fp: TextLoader(str(fp), encoding="utf-8"),
    ".pdf":  lambda fp: OCRPDFLoader(str(fp)),
    ".docx": lambda fp: OCRDOCLoader(str(fp)),
    ".md":   lambda fp: UnstructuredMarkdownLoader(str(fp)),
}


def load_documents_from_directory(directory_path: str) -> list:
    """
    从指定目录递归加载所有支持的文档，返回 Document 列表。
    自动从目录名提取领域标签（如 "banking_data" → "banking"）。
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        logger.warning(f"目录不存在，跳过: {directory_path}")
        return []

    source = dir_path.name.replace("_data", "")
    documents = []
    supported_exts = set(DOCUMENT_LOADERS.keys())

    for root, _, files in os.walk(dir_path):
        for fname in files:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            if ext not in supported_exts:
                logger.debug(f"不支持的文件类型，跳过: {fpath}")
                continue
            try:
                loader = DOCUMENT_LOADERS[ext](fpath)
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = source
                    doc.metadata["file_path"] = str(fpath)
                    doc.metadata["timestamp"] = datetime.now().isoformat()
                documents.extend(loaded)
                logger.info(f"加载文件: {fpath} ({len(loaded)} 段)")
            except Exception as e:
                logger.error(f"加载文件失败 {fpath}: {e}")

    logger.info(f"目录 {directory_path} 共加载 {len(documents)} 段文档")
    return documents


def process_documents(
    directory_path: str,
    parent_chunk_size: int = None,
    child_chunk_size: int = None,
    chunk_overlap: int = None,
    doc_id: int = None,
) -> list:
    """
    文档加载 + 双层切分，返回子块列表。

    父块切分：使用 ChineseRecursiveTextSplitter（或 MarkdownTextSplitter 处理 .md）
    子块切分：使用 ChineseRecursiveTextSplitter，父块内容写入 metadata["parent_content"]
    """
    cfg = get_config()
    parent_chunk_size = parent_chunk_size or cfg.retrieval.parent_chunk_size
    child_chunk_size  = child_chunk_size  or cfg.retrieval.child_chunk_size
    chunk_overlap     = chunk_overlap     or cfg.retrieval.chunk_overlap

    documents = load_documents_from_directory(directory_path)
    if not documents:
        logger.warning("无文档可处理")
        return []

    # 初始化切分器
    parent_splitter = ChineseRecursiveTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
    )
    child_splitter = ChineseRecursiveTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
    )
    md_parent_splitter = MarkdownTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
    )
    md_child_splitter = MarkdownTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
    )

    child_chunks = []
    parent_idx = 0  # 父块全局序号（跨所有子文档自增，保证 chunk id 唯一）
    for i, doc in enumerate(documents):
        ext = Path(doc.metadata.get("file_path", "")).suffix.lower()
        is_md = (ext == ".md")
        p_splitter = md_parent_splitter if is_md else parent_splitter
        c_splitter = md_child_splitter  if is_md else child_splitter

        parent_docs = p_splitter.split_documents([doc])
        logger.debug(f"文档 {i}: 父块 {len(parent_docs)} 个")

        # chunk id 必须全局唯一：单文件上传时文件序号 i 恒为 0，若只用 i 会导致
        # 同知识库多文件产生相同 chunk id（检索去重按 question 会丢块/串内容）。
        # 上传路径传入全局唯一的 doc_id，保证每块 id 唯一；目录批量导入无单一 doc_id 时回退用 i。
        # 父块索引必须跨「全部加载出的子文档（如 PDF 每一页）」全局自增，
        # 否则多页 PDF 每页都从 parent_0 重新计数，导致 doc_31_parent_0_child_0
        # 在每一页重复出现，chunk id 大规模互撞、检索去重丢块。
        id_prefix = f"doc_{doc_id}" if doc_id is not None else f"doc_{i}"
        for j, parent_doc in enumerate(parent_docs):
            parent_id = f"{id_prefix}_parent_{parent_idx}"
            sub_chunks = c_splitter.split_documents([parent_doc])
            for k, sub in enumerate(sub_chunks):
                sub.metadata["parent_id"]   = parent_id
                sub.metadata["parent_content"] = parent_doc.page_content
                sub.metadata["id"]          = f"{parent_id}_child_{k}"
                child_chunks.append(sub)
            parent_idx += 1

    logger.info(f"双层切分完成：父块 → 子块，共 {len(child_chunks)} 个子块")
    return child_chunks
