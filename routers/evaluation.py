#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
routers/evaluation.py — RAG 评估路由
提供评估数据集生成、RAG 系统评估、结果查看等接口（管理员专属）。
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from loguru import logger

from models import ApiResponse
from routers.auth import get_admin_user
from services.evaluation import (
    generate_evaluation_dataset,
    save_dataset,
    load_dataset,
    list_datasets,
    run_rag_on_dataset,
    evaluate_with_ragas,
    run_full_evaluation,
    list_experiments,
    get_experiment_detail,
    compare_experiments,
)

router = APIRouter(prefix="/evaluation", tags=["RAG评估"])


# ─── 请求模型 ──────────────────────────────────────────────────────────────────
class GenerateDatasetRequest(BaseModel):
    num_seeds: int = 10
    questions_per_seed: int = 3
    dataset_name: str = "auto_generated"


class RunEvaluationRequest(BaseModel):
    num_seeds: int = 10
    questions_per_seed: int = 3


class EvaluateDatasetRequest(BaseModel):
    dataset_name: str = "auto_generated"


# ─── 评估数据集管理 ────────────────────────────────────────────────────────────

@router.post("/generate-dataset", response_model=ApiResponse, summary="LLM 生成评估数据集")
def generate_dataset(
    req: GenerateDatasetRequest,
    current_user: dict = Depends(get_admin_user),
):
    """使用 LLM 从现有 FAQ 数据生成多样化的评估数据集（管理员专属）。"""
    logger.info(f"管理员 {current_user['username']} 触发评估数据集生成")
    dataset = generate_evaluation_dataset(
        num_seeds=req.num_seeds,
        questions_per_seed=req.questions_per_seed,
    )
    if not dataset:
        raise HTTPException(status_code=500, detail="数据集生成失败，请检查 FAQ 数据和 LLM 配置")

    file_path = save_dataset(dataset, req.dataset_name)
    return ApiResponse(success=True, data={
        "num_questions": len(dataset),
        "file_path": file_path,
        "sample_questions": [d["question"] for d in dataset[:5]],
    })


@router.get("/datasets", response_model=ApiResponse, summary="列出所有评估数据集")
def get_datasets(current_user: dict = Depends(get_admin_user)):
    """列出所有已保存的评估数据集（管理员专属）。"""
    datasets = list_datasets()
    return ApiResponse(success=True, data=datasets)


@router.get("/datasets/{name}", response_model=ApiResponse, summary="加载指定数据集")
def get_dataset(
    name: str,
    current_user: dict = Depends(get_admin_user),
):
    """加载指定名称的数据集内容（管理员专属）。"""
    dataset = load_dataset(name)
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return ApiResponse(success=True, data={
        "num_questions": len(dataset),
        "questions": dataset,
    })


# ─── RAG 评估 ──────────────────────────────────────────────────────────────────

@router.post("/run", response_model=ApiResponse, summary="一键运行完整 RAG 评估")
def run_evaluation(
    req: RunEvaluationRequest,
    current_user: dict = Depends(get_admin_user),
):
    """
    一键完成完整评估流程：LLM 生成数据集 → RAG 管线收集响应 → RAGAS 评估。
    注意：此接口执行时间较长（取决于问题数量和 LLM 响应速度）。
    """
    logger.info(f"管理员 {current_user['username']} 启动完整 RAG 评估")
    results = run_full_evaluation(
        num_seeds=req.num_seeds,
        questions_per_seed=req.questions_per_seed,
    )
    if "error" in results:
        raise HTTPException(status_code=500, detail=results["error"])
    return ApiResponse(success=True, data=results)


@router.post("/evaluate", response_model=ApiResponse, summary="对已有数据集运行 RAGAS 评估")
def evaluate_existing_dataset(
    req: EvaluateDatasetRequest,
    current_user: dict = Depends(get_admin_user),
):
    """对已存在的数据集运行 RAG 管线 + RAGAS 评估（管理员专属）。"""
    dataset = load_dataset(req.dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"数据集 {req.dataset_name} 不存在")

    logger.info(f"管理员 {current_user['username']} 对数据集 {req.dataset_name} 运行评估")

    # Step 1: 运行 RAG 管线
    eval_data = run_rag_on_dataset(dataset)

    # Step 2: RAGAS 评估
    results = evaluate_with_ragas(eval_data)
    if "error" in results:
        raise HTTPException(status_code=500, detail=results["error"])

    return ApiResponse(success=True, data=results)


# ─── 评估结果查看 ──────────────────────────────────────────────────────────────

# 注意：/experiments/compare 必须注册在 /experiments/{filename} 之前，否则被路径参数抢占
@router.get("/experiments/compare", response_model=ApiResponse, summary="对比多个评估实验指标")
def compare_experiments_api(
    files: str = Query("", description="逗号分隔的实验文件名；为空时取最近 N 次实验"),
    latest: int = Query(2, ge=2, le=5, description="files 为空时取最近 N 次实验对比"),
    current_user: dict = Depends(get_admin_user),
):
    """对比多个评估实验的 RAGAS 指标（逐指标给出各实验取值、最优实验与极差），
    支撑「调整检索/Prompt 配置 → 跑评估 → 对比实验」的效果迭代闭环。"""
    if files.strip():
        filenames = [f.strip() for f in files.split(",") if f.strip()]
    else:
        recent = list_experiments()[:latest]
        if len(recent) < 2:
            raise HTTPException(status_code=400, detail="历史实验不足两次，请先运行评估生成实验记录")
        filenames = [e["filename"] for e in recent]

    result = compare_experiments(filenames)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return ApiResponse(success=True, data=result)


@router.get("/experiments", response_model=ApiResponse, summary="列出所有评估实验")
def get_experiments(current_user: dict = Depends(get_admin_user)):
    """列出所有历史评估实验记录（管理员专属）。"""
    experiments = list_experiments()
    return ApiResponse(success=True, data=experiments)


@router.get("/experiments/{filename}", response_model=ApiResponse, summary="获取实验详情")
def get_experiment(
    filename: str,
    current_user: dict = Depends(get_admin_user),
):
    """获取指定评估实验的详细结果（管理员专属）。"""
    detail = get_experiment_detail(filename)
    if not detail:
        raise HTTPException(status_code=404, detail="实验记录不存在")
    return ApiResponse(success=True, data=detail)
