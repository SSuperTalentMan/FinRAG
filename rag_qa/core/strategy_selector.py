#!/usr/bin/env python
"""
rag_qa/core/strategy_selector.py — 检索策略选择器

两级决策（控制延迟）：
  1. 规则预筛（select_strategy_by_rules）：零成本零延迟，基于信号词与查询长度路由；
     绝大多数意图明确的查询直接命中"直接检索"，不产生任何额外开销。
  2. LLM 选择器（StrategySelector）：仅当开启 llm_fallback 且规则未命中长查询时调用，
     失败降级为直接检索。
"""

from loguru import logger

from .prompts import RAGPrompts

# ─── 策略常量 ──────────────────────────────────────────────────────────────────
STRATEGY_DIRECT    = "直接检索"
STRATEGY_HYDE      = "假设问题检索"
STRATEGY_SUBQUERY  = "子查询检索"
STRATEGY_BACKTRACK = "回溯问题检索"

ALL_STRATEGIES = [STRATEGY_DIRECT, STRATEGY_HYDE, STRATEGY_SUBQUERY, STRATEGY_BACKTRACK]

# ─── 规则预筛信号词（保守设计：宁可漏选也不误路由，未命中一律直接检索）────────────
# 对比类：涉及多个实体/方面，拆分后分别检索再合并效果更好
_COMPARISON_MARKERS = (
    "对比", "比较", "区别", "差异", "不同", "哪个好", "哪个更", "优缺点",
    "分别", "还是", "vs", "VS",
)
# 抽象/开放类：直接向量检索对宽泛表述召回差，先生成假设答案再检索（HyDE）
_ABSTRACT_MARKERS = (
    "有哪些", "影响", "意义", "作用", "前景", "发展趋势", "怎么看", "如何看待",
    "为什么", "意味着", "说明什么",
)
# 场景/复杂类：前置条件多，先简化为核心问题再检索
_SCENARIO_MARKERS = (
    "我有一个", "我们公司", "假设", "如果我", "怎么办", "如何处理", "该怎么",
)

# 规则触发所需的最短查询长度（短查询意图通常明确，直接检索即可）
_MIN_LEN_FOR_HYDE = 15
_MIN_LEN_FOR_BACKTRACK = 20
# LLM 兜底生效的最短查询长度（更短的查询 LLM 也判不出增益）
_MIN_LEN_FOR_LLM_FALLBACK = 20


def select_strategy_by_rules(query: str) -> str:
    """基于信号词与长度的规则路由（零成本）。

    匹配优先级：对比类 > 场景类 > 抽象类；未命中一律返回直接检索。
    """
    q = (query or "").strip()
    if not q:
        return STRATEGY_DIRECT
    if any(m in q for m in _COMPARISON_MARKERS):
        return STRATEGY_SUBQUERY
    if any(m in q for m in _SCENARIO_MARKERS) and len(q) >= _MIN_LEN_FOR_BACKTRACK:
        return STRATEGY_BACKTRACK
    if any(m in q for m in _ABSTRACT_MARKERS) and len(q) >= _MIN_LEN_FOR_HYDE:
        return STRATEGY_HYDE
    return STRATEGY_DIRECT


class StrategySelector:
    """LLM 驱动的多策略检索选择器（慢路径，仅在规则未命中且开启兜底时使用）。"""

    def select_strategy(self, query: str) -> str:
        """
        调用 LLM 为查询选择最合适的检索策略。
        返回策略名称字符串，失败时降级为 STRATEGY_DIRECT。
        """
        # 延迟导入：打破 rag_qa.core.__init__ → strategy_selector → services.llm 的
        # 导入环（services.llm 顶层还会反向导入 rag_qa.core.prompts）
        from services.llm import chat_completion
        try:
            prompt = RAGPrompts.strategy_select_prompt(query)
            result = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            strategy = result.strip()
            # 模糊匹配容错
            for s in ALL_STRATEGIES:
                if s in strategy or strategy in s:
                    logger.debug(f"策略选择器: '{query}' → {s}")
                    return s
            logger.warning(f"策略选择器返回未知策略: {strategy}，降级为直接检索")
            return STRATEGY_DIRECT
        except Exception as e:
            logger.error(f"策略选择器调用失败: {e}，降级为直接检索")
            return STRATEGY_DIRECT


def select_strategy(query: str, llm_fallback: bool = False) -> str:
    """检索策略统一入口：规则预筛优先，可选 LLM 兜底。

    llm_fallback=True 时，规则未命中（返回直接检索）且查询足够长的场景
    会再尝试一次 LLM 选择；其余情况直接返回规则结果，不产生 LLM 开销。
    """
    by_rules = select_strategy_by_rules(query)
    if by_rules != STRATEGY_DIRECT or not llm_fallback:
        return by_rules
    if len((query or "").strip()) < _MIN_LEN_FOR_LLM_FALLBACK:
        return by_rules
    logger.debug(f"规则未命中，LLM 兜底选择策略: {query[:40]}...")
    return StrategySelector().select_strategy(query)
