#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/init_multimodal.py — 初始化多模态融合（P1）数据层：建表 + 种子监管规则。

用法：
    python scripts/init_multimodal.py            # 建表（幂等）
    python scripts/init_multimodal.py --rebuild  # 建表 + 覆盖写入种子监管规则
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_qa.review import db as rdb  # noqa: E402
from db.mysql import get_db  # noqa: E402

# 种子监管规则：保险/金融合规 + 通用商事合同风险条款（rule_id / 来源 / 条款号 / 要求 / 严重度 / 关键词）
_SEED_RULES = [
    ("RULE_001", "商业银行代理保险业务管理办法", "第十五条", "保险公司应当在投保人填写投保单前，向投保人出示保险条款，并说明犹豫期、退保等重要事项，不得误导投保人。", "high", "投保单 保险条款 犹豫期 退保"),
    ("RULE_002", "商业银行代理保险业务管理办法", "第二十七条", "严禁销售人员承诺不确定的收益，严禁使用'稳赚''保本保息'等误导性用语。", "high", "收益 承诺 稳赚 保本 误导"),
    ("RULE_003", "人身保险产品信息披露管理办法", "第九条", "产品说明书应当完整载明犹豫期、退保损失、等待期、免责条款等关键信息，不得隐瞒。", "medium", "披露 犹豫期 退保损失 等待期 免责"),
    ("RULE_004", "保险法", "第四十七条", "投保人解除人身保险合同的，保险人应当自收到解除通知之日起三十日内退还保险单的现金价值。", "medium", "解除 退保 三十日 现金价值"),
    ("RULE_005", "保险法", "第三十二条", "投保人对被保险人应当具有保险利益，不得为不具有保险利益的人投保。", "medium", "保险利益 投保"),
    ("RULE_006", "人身保险产品信息披露管理办法", "第十四条", "保险公司不得对保险产品作虚假或者引人误解的宣传，不得夸大保险责任。", "medium", "虚假 宣传 夸大 保险责任"),
    ("RULE_007", "偿付能力管理规定", "第三十六条", "保险公司应当在保单上如实载明犹豫期及犹豫期内投保人可以无条件解除合同的权利。", "low", "犹豫期 无条件 解除"),
    ("RULE_008", "销售行为可回溯管理规定", "第六条", "保险公司应当对保险销售过程进行录音录像，并完整保存销售记录。", "medium", "录音录像 销售过程 保存"),
    # —— 通用商事合同风险条款（补全 docaudit_risk_rule，避免审查时 insufficient_evidence 退化）——
    ("RULE_101", "民法典·合同编", "第五百六十三条", "合同解除权应双方对等；一方随时单方解除合同且不承担任何赔偿或违约责任，构成权利义务严重不对等。", "high", "单方解除 随时解除 不承担责任 不赔偿 解除权"),
    ("RULE_102", "民法典·合同编", "第五百四十三条", "合同变更须当事人协商一致；一方有权单方解释并随时修改合同条款且对方不得提出异议，剥夺对方协商权，显失公平。", "high", "单方解释 随时修改 不得异议 修改合同 解释权"),
    ("RULE_103", "民法典·合同编", "第五百九十二条", "当事人一方违约应赔偿对方损失；要求一方放弃向对方主张违约金或损害赔偿的权利，免除自身主要责任，无效且显失公平。", "high", "放弃索赔 放弃违约金 损害赔偿 权利放弃 索赔权"),
    ("RULE_104", "民法典·合同编", "第五百零六条", "造成对方人身损害或因故意、重大过失造成对方财产损失的免责条款无效；一方完全免除自身责任无效。", "high", "免除责任 免责 不承担责任 责任豁免 免责条款"),
    ("RULE_105", "民法典·合同编", "第五百六十五条", "合同终止/续展应明确并保障当事人退出权；到期自动续展/续期仅以提前通知为条件且未给予充分提醒与明确退出机制，损害对方选择权。", "medium", "自动续展 自动续期 到期续展 续展 续期"),
    ("RULE_106", "民法典·合同编", "第五百八十五条", "违约金由当事人约定，过分高于造成损失的可请求适当减少；双方对等违约金属正常商业安排，应明示计算标准。", "low", "违约金 罚金 违约责任 赔偿 违约金标准"),
    ("RULE_107", "反不正当竞争法", "第九条", "商业秘密保密义务及对应违约罚则属正常合同安排；但保密范围与罚则应合理，不得显失公平。", "low", "保密义务 商业秘密 保密 违约罚则"),
    ("RULE_108", "民事诉讼法", "第三十五条", "协议管辖/适用法应明确且不排除对方主要诉讼权利；约定一方所在地法院管辖属常见，一般不单独构成违规。", "low", "管辖 适用法 争议解决 法院 诉讼"),
]

_TABLES = ["docaudit_document", "docaudit_page_unit", "docaudit_clause",
           "docaudit_risk_rule", "docaudit_review_task", "docaudit_clause_review", "docaudit_report"]


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    rdb.ensure_tables()
    print("[init_multimodal] 审查相关表已就绪:", ", ".join(_TABLES))

    if rebuild:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM docaudit_risk_rule")
    # 幂等 upsert：新增规则（如通用商事合同 RULE_101+）会在非 rebuild 时也写入，
    # 已有规则按 requirement 更新；不再因表非空而整体跳过，避免规则库永远停留在旧集合。
    with get_db() as conn:
        cur = conn.cursor()
        for rid, doc, article, req, sev, kw in _SEED_RULES:
            cur.execute(
                "INSERT INTO docaudit_risk_rule (rule_id, doc_name, article_no, requirement, severity, keywords) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE requirement=VALUES(requirement), severity=VALUES(severity), "
                "keywords=VALUES(keywords), doc_name=VALUES(doc_name), article_no=VALUES(article_no)",
                (rid, doc, article, req, sev, kw),
            )
    print(f"[init_multimodal] 种子监管规则写入完成（{len(_SEED_RULES)} 条，rebuild={rebuild}）")


if __name__ == "__main__":
    main()