#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services/evaluation.py — RAG 项目评估服务（RAGAS 框架）
通过 LLM 从现有 FAQ 数据生成评估数据集，运行 RAG 管线收集响应，使用 RAGAS 指标评估。
"""

import json
import time
import random
import os
import sys
import types
from pathlib import Path
from typing import Optional

from loguru import logger

from config import get_config
from db.mysql import get_all_qa_for_bm25
from services.llm import get_llm_client, chat_completion


# ─── RAGAS 兼容性修复 ──────────────────────────────────────────────────────────
# langchain_community 0.4.x 移除了 vertexai 模块，但 RAGAS 仍引用它。
# 创建一个空的 stub 模块以避免 ImportError。
def _patch_langchain_vertexai():
    try:
        from langchain_community.chat_models import vertexai  # noqa: F401
    except ImportError:
        stub = types.ModuleType("langchain_community.chat_models.vertexai")
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = stub
        setattr(
            __import__("langchain_community.chat_models", fromlist=[""]),
            "vertexai",
            stub,
        )

_patch_langchain_vertexai()


from services.bm25 import get_bm25_retriever
from services.embedding import encode_query_dense_sparse
from db.milvus import get_milvus_client, search_milvus, ensure_collection
from services.reranker import rerank
from rag_qa.core.query_classifier import classify_intent


# ─── 评估数据存储 ──────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parent.parent / "evals"
DATASETS_DIR = EVAL_DIR / "datasets"
EXPERIMENTS_DIR = EVAL_DIR / "experiments"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. LLM 生成评估数据集 ────────────────────────────────────────────────────

def _load_faq_samples(num: int = 50) -> list[dict]:
    """从 MySQL 和本地 JSONL 文件加载 FAQ 样本作为生成种子。"""
    samples = []

    # 从 MySQL 加载
    try:
        all_qa = get_all_qa_for_bm25()
        random.shuffle(all_qa)
        samples.extend(all_qa[:num])
    except Exception as e:
        logger.warning(f"从 MySQL 加载 FAQ 失败: {e}")

    # 从本地 JSONL 文件补充
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if len(samples) < num:
        jsonl_files = list(data_dir.rglob("*.jsonl"))
        random.shuffle(jsonl_files)
        for f in jsonl_files:
            if len(samples) >= num:
                break
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if item.get("question") and item.get("answer"):
                            samples.append(item)
                        if len(samples) >= num:
                            break
            except Exception as e:
                logger.debug(f"读取 {f} 失败: {e}")

    random.shuffle(samples)
    return samples[:num]


def _generate_questions_batch(seed_qa: list[dict], batch_size: int = 5) -> list[dict]:
    """
    使用 LLM 从种子 Q&A 生成多样化的评估问题。
    每条种子生成 3 种类型的问题：事实性、推理型、变换表述。
    """
    seed_text = "\n".join(
        f"{i+1}. 问题: {q.get('question', '')}\n   答案: {q.get('answer', '')[:100]}"
        for i, q in enumerate(seed_qa)
    )

    prompt = f"""你是一个金融领域测试数据生成专家。请基于以下种子问答对，为每一条生成 3 个新的评估问题。

种子问答：
{seed_text}

生成要求：
1. 对每条种子生成 3 种类型的问题：
   - 事实性：直接询问事实，答案明确
   - 推理型：需要基于种子信息进行推理
   - 变换表述：用不同的措辞重述原问题
2. 新问题不能与种子问题完全相同
3. 必须为每个新问题提供参考答案（基于种子答案）
4. 标注领域类别（banking / corporate_finance / financial_accounting 等）

请以 JSON 数组格式输出，每个元素格式：
{{"question": "新问题", "reference": "参考答案", "type": "事实性|推理型|变换表述", "domain": "领域"}}

只输出 JSON 数组，不要其他文字："""

    messages = [{"role": "user", "content": prompt}]
    raw = chat_completion(messages, stream=False)

    # 解析 LLM 返回的 JSON
    try:
        # 尝试提取 JSON 数组
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            json_str = raw[start:end]
            items = json.loads(json_str)
            return items
    except json.JSONDecodeError as e:
        logger.error(f"解析 LLM 生成的评估数据失败: {e}")

    return []


def generate_evaluation_dataset(
    num_seeds: int = 10,
    questions_per_seed: int = 3,
) -> list[dict]:
    """
    使用 LLM 从现有 FAQ 数据生成评估数据集。
    返回格式：[{"question", "reference", "type", "domain"}]
    """
    logger.info(f"开始生成评估数据集: {num_seeds} 条种子, 每条生成 {questions_per_seed} 个问题")

    seeds = _load_faq_samples(num_seeds)
    if not seeds:
        logger.error("无法加载 FAQ 种子数据")
        return []

    all_questions = []

    # 分批处理，每批 5 条种子
    batch_size = 5
    for i in range(0, len(seeds), batch_size):
        batch = seeds[i:i + batch_size]
        logger.info(f"处理批次 {i // batch_size + 1}/{(len(seeds) + batch_size - 1) // batch_size}")
        generated = _generate_questions_batch(batch)
        all_questions.extend(generated)
        time.sleep(0.5)  # 避免请求过快

    # 去重
    seen = set()
    unique_questions = []
    for q in all_questions:
        question = q.get("question", "").strip()
        if question and question not in seen:
            seen.add(question)
            unique_questions.append({
                "question": question,
                "reference": q.get("reference", ""),
                "type": q.get("type", "事实性"),
                "domain": q.get("domain", "banking"),
            })

    logger.info(f"评估数据集生成完成: {len(unique_questions)} 个唯一问题")
    return unique_questions


def save_dataset(dataset: list[dict], name: str = "default") -> str:
    """保存评估数据集到文件。"""
    file_path = DATASETS_DIR / f"{name}_{int(time.time())}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"数据集已保存: {file_path}")
    return str(file_path)


def load_dataset(name: str) -> list[dict]:
    """加载指定名称的数据集。"""
    files = sorted(DATASETS_DIR.glob(f"{name}_*.json"), reverse=True)
    if not files:
        return []
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def list_datasets() -> list[dict]:
    """列出所有可用的评估数据集。"""
    result = []
    for f in sorted(DATASETS_DIR.glob("*.json"), reverse=True):
        stat = f.stat()
        result.append({
            "filename": f.name,
            "size": stat.st_size,
            "created_at": stat.st_mtime,
        })
    return result


# ─── 2. 运行 RAG 管线收集响应 ──────────────────────────────────────────────────

def _build_context_for_eval(question: str) -> tuple[str, list[str]]:
    """
    执行 RAG 检索管线，直接复用 chat.py 的生产管线（routers.chat._build_context），
    避免第三份重复实现导致评估与生产行为漂移。返回 (context_str, contexts_list)，
    contexts_list 为 RAGAS 所需的 List[str]（与喂给 LLM 的上下文逐段对齐）。
    """
    from routers.chat import _build_context
    try:
        intent = classify_intent(question)
        context, _sources = _build_context(question, intent)
        contexts = [p for p in context.split("\n\n") if p.strip()] if context else []
        return context, contexts
    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")
        return "", []


def _generate_answer_for_eval(question: str, context: str, domain: str) -> str:
    """调用 LLM 生成回答（非流式）。

    Prompt 复用生产管线的 RAGPrompts.answer_messages，保证评估环境与
    线上行为一致（否则评估分数无法反映真实服务的回答质量）。
    """
    if not context:
        # 无检索结果时直接让 LLM 回答
        return chat_completion([{"role": "user", "content": question}], stream=False)

    from rag_qa.core.prompts import RAGPrompts
    messages = RAGPrompts.answer_messages(question, context, domain)
    return chat_completion(messages, stream=False)


def run_rag_on_dataset(dataset: list[dict]) -> list[dict]:
    """
    对数据集中的每个问题运行 RAG 管线，收集响应和检索上下文。
    返回 RAGAS 格式的评估数据。
    """
    results = []
    total = len(dataset)

    for i, item in enumerate(dataset):
        question = item["question"]
        reference = item.get("reference", "")
        domain = item.get("domain", "general")

        logger.info(f"RAG 评估 [{i+1}/{total}]: {question[:50]}...")

        # 1. 检索
        context, contexts = _build_context_for_eval(question)

        # 2. 生成回答
        answer = _generate_answer_for_eval(question, context, domain)

        # 3. 组装 RAGAS 评估样本
        results.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": reference,
        })

        time.sleep(0.3)  # 避免 LLM 请求过快

    return results


# ─── 3. RAGAS 评估 ─────────────────────────────────────────────────────────────

def _get_ragas_llm():
    """创建 RAGAS 评估用的 LLM 包装器（适配 ragas 0.3.x）。

    ragas 0.3.x 的 llm_factory(model, base_url=...) 内部构造 ChatOpenAI，
    不再接受 provider/client 参数；API Key 通过 OPENAI_API_KEY 环境变量
    注入（langchain 客户端约定），base_url 指向 DashScope 兼容端点。
    """
    from ragas.llms import llm_factory

    cfg = get_config()
    if cfg.api_key:
        os.environ.setdefault("OPENAI_API_KEY", cfg.api_key)
    return llm_factory(cfg.llm.model, base_url=cfg.llm.base_url)


def evaluate_with_ragas(eval_data: list[dict]) -> dict:
    """
    使用 RAGAS 框架评估 RAG 系统。
    指标：Faithfulness（忠实度）、LLMContextRecall（上下文召回率）、FactualCorrectness（事实正确性）
    """
    try:
        from ragas import evaluate, EvaluationDataset
        try:
            from ragas.metrics.collections import (
                LLMContextRecall,
                Faithfulness,
                FactualCorrectness,
                AnswerRelevancy,
            )
        except ImportError:
            from ragas.metrics import (
                LLMContextRecall,
                Faithfulness,
                FactualCorrectness,
                AnswerRelevancy,
            )
    except ImportError as e:
        logger.error(f"RAGAS 未安装: {e}")
        return {"error": "RAGAS 未安装，请运行: pip install ragas", "detail": str(e)}

    if not eval_data:
        return {"error": "评估数据为空"}

    logger.info(f"开始 RAGAS 评估，共 {len(eval_data)} 条样本")

    # 转换为 RAGAS EvaluationDataset
    dataset = EvaluationDataset.from_list(eval_data)

    # 获取评估用 LLM
    evaluator_llm = _get_ragas_llm()

    # 运行评估
    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        AnswerRelevancy(),
    ]

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
        )
    except Exception as e:
        logger.error(f"RAGAS 评估失败: {e}")
        # 尝试只用部分指标
        logger.info("尝试仅使用 Faithfulness 和 LLMContextRecall 指标...")
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[LLMContextRecall(), Faithfulness()],
                llm=evaluator_llm,
            )
        except Exception as e2:
            logger.error(f"RAGAS 评估仍然失败: {e2}")
            return {"error": str(e2)}

    # 提取结果
    results_dict = {}
    try:
        # result 是一个 dict-like 对象
        for k, v in result.items():
            if isinstance(v, (int, float)):
                results_dict[k] = round(v, 4)
            else:
                results_dict[k] = v
    except Exception:
        results_dict = {"raw": str(result)}

    # 保存详细结果
    detail_rows = []
    try:
        # 尝试获取每条样本的分数
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            for _, row in df.iterrows():
                detail_rows.append({
                    "question": str(row.get("user_input", "")),
                    "response": str(row.get("response", ""))[:200],
                    "faithfulness": float(row.get("faithfulness", 0)) if row.get("faithfulness") is not None else None,
                    "context_recall": float(row.get("context_recall", 0)) if row.get("context_recall") is not None else None,
                    "factual_correctness": float(row.get("factual_correctness", 0)) if row.get("factual_correctness") is not None else None,
                    "answer_relevancy": float(row.get("answer_relevancy", 0)) if row.get("answer_relevancy") is not None else None,
                })
    except Exception as e:
        logger.warning(f"获取详细结果失败: {e}")

    output = {
        "summary": results_dict,
        "num_samples": len(eval_data),
        "details": detail_rows,
        "evaluated_at": int(time.time()),
    }

    # 保存到文件
    result_file = EXPERIMENTS_DIR / f"eval_{int(time.time())}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存: {result_file}")

    return output


def list_experiments() -> list[dict]:
    """列出所有历史评估实验。"""
    result = []
    for f in sorted(EXPERIMENTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            result.append({
                "filename": f.name,
                "evaluated_at": data.get("evaluated_at", 0),
                "num_samples": data.get("num_samples", 0),
                "summary": data.get("summary", {}),
            })
        except Exception:
            continue
    return result


def get_experiment_detail(filename: str) -> Optional[dict]:
    """获取指定实验的详细信息。"""
    file_path = EXPERIMENTS_DIR / filename
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_experiments(filenames: list[str]) -> dict:
    """对比多个评估实验的 RAGAS 指标，支撑"调参 → 评估 → 对比"的迭代闭环。

    filenames 为 EXPERIMENTS_DIR 下的实验文件名（list_experiments 返回的 filename）。
    旧实验文件可能缺少部分指标，缺失值不参与 best/delta 计算。
    """
    if not filenames or len(filenames) < 2:
        return {"error": "至少需要两个实验文件才能对比"}

    experiments = []
    for fn in filenames:
        detail = get_experiment_detail(fn)
        if not detail:
            return {"error": f"实验记录不存在: {fn}"}
        experiments.append({
            "filename": fn,
            "evaluated_at": detail.get("evaluated_at", 0),
            "num_samples": detail.get("num_samples", 0),
            "summary": detail.get("summary", {}),
        })

    # 逐指标对比：各实验的取值、最优实验、极差
    metric_names = sorted({
        k for e in experiments for k, v in e["summary"].items() if isinstance(v, (int, float))
    })
    metrics: dict[str, dict] = {}
    for name in metric_names:
        values = {e["filename"]: e["summary"].get(name) for e in experiments}
        nums = [v for v in values.values() if isinstance(v, (int, float))]
        metrics[name] = {
            "values": values,
            "best": max(values, key=lambda k: values[k]) if nums else None,
            "delta": round(max(nums) - min(nums), 4) if len(nums) >= 2 else None,
        }
    return {"experiments": experiments, "metrics": metrics}


# ─── 4. 一键完整评估流程 ──────────────────────────────────────────────────────

def run_full_evaluation(
    num_seeds: int = 10,
    questions_per_seed: int = 3,
) -> dict:
    """
    一键完成完整评估流程：
    1. LLM 生成评估数据集
    2. 运行 RAG 管线收集响应
    3. RAGAS 评估
    """
    start_time = time.time()

    # Step 1: 生成评估数据集
    logger.info("=" * 50)
    logger.info("Step 1: LLM 生成评估数据集")
    logger.info("=" * 50)
    dataset = generate_evaluation_dataset(num_seeds, questions_per_seed)
    if not dataset:
        return {"error": "评估数据集生成失败"}

    save_dataset(dataset, "auto_generated")

    # Step 2: 运行 RAG 管线
    logger.info("=" * 50)
    logger.info(f"Step 2: 运行 RAG 管线收集响应 ({len(dataset)} 个问题)")
    logger.info("=" * 50)
    eval_data = run_rag_on_dataset(dataset)

    # Step 3: RAGAS 评估
    logger.info("=" * 50)
    logger.info("Step 3: RAGAS 框架评估")
    logger.info("=" * 50)
    results = evaluate_with_ragas(eval_data)

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 1)
    results["dataset_size"] = len(dataset)

    logger.info(f"评估完成，耗时 {elapsed:.1f}s")
    return results
