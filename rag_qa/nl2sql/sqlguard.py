#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql/sqlguard.py — SQLGuard：sqlglot AST 静态校验（纵深防御第二层）。

忠实移植自 ChatBI 的 app/guard/sqlguard.py，仅将配置来源切换到 FinRag 的 get_config().nl2sql。

规则：
1. 仅允许单条 SELECT / UNION（含 CTE）；任何 DML/DDL/命令/INTO 一律拒绝；
2. 表名白名单：解析出的物理表必须在允许集合内，杜绝幻觉表名与越权；跨库访问（非业务库前缀）拒绝；
3. 危险函数黑名单（SLEEP/BENCHMARK/LOAD_FILE/锁函数）与系统变量（@@）；
4. 禁止 INTO OUTFILE/DUMPFILE、多语句、系统库、MySQL 版本注释（/*!...*/）绕过；
5. 强制 LIMIT：缺失补默认值、超上限改写；注释剥离后基于 AST 规范化输出；
6. 规范化后二次校验：重新解析确认语句类型未变（防"规范化改变语义"类绕过，如 SELECT...INTO→CREATE TABLE AS）。
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from config import get_config
from rag_qa.nl2sql.schema import GuardResult

DENY_FUNCTIONS = {
    "SLEEP", "BENCHMARK", "LOAD_FILE", "GET_LOCK", "RELEASE_LOCK",
    "IS_FREE_LOCK", "IS_USED_LOCK", "RELEASE_ALL_LOCKS",
    "MASTER_POS_WAIT", "SOURCE_POS_WAIT",
}

RE_DANGEROUS_TEXT = re.compile(
    r"into\s+(outfile|dumpfile)|@@|information_schema|performance_schema|\bmysql\.[a-z_]+|\bsys\.[a-z_]+",
    re.IGNORECASE,
)
RE_MULTI_SEMI = re.compile(r";\s*\S")
RE_VERSION_COMMENT = re.compile(r"/\*!")
RE_STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.|\"\")*\"")


def _strip_literals(sql: str) -> str:
    """掏出字符串字面量内容，只保留 SQL 结构。

    这样 `WHERE city='北京;朝阳'` 里的分号不会误判成多语句，
    而真正的 `SELECT 1; DROP TABLE x` 依然被拦下（分号在结构里）。
    """
    return RE_STRING_LITERAL.sub("''", sql)


def _query_types() -> tuple[type, ...]:
    return exp.Select, exp.Union


def _mask_literals(stmt: exp.Expression) -> str:
    """把 AST 中字符串字面量替换为空串后再渲染，避免危险关键字正则误杀业务数据。"""
    try:
        masked = stmt.transform(
            lambda n: exp.Literal.string("") if isinstance(n, exp.Literal) and n.is_string else n,
            copy=True,
        )
        return masked.sql(dialect="mysql", comments=False)
    except Exception:
        return stmt.sql(dialect="mysql", comments=False)


def _iter_deny_nodes(stmt: exp.Expression):
    deny_names = (
        "Insert", "Update", "Delete", "Merge", "Drop", "Alter", "AlterTable",
        "Create", "TruncateTable", "Grant", "Revoke", "Use", "Set", "Command",
        "Into",
    )
    deny_types = tuple(
        cls for cls in (getattr(exp, n, None) for n in deny_names)
        if isinstance(cls, type) and issubclass(cls, exp.Expression)
    )
    for node in stmt.walk():
        if isinstance(node, deny_types):
            yield "包含不允许的语句类型(DML/DDL/命令/INTO)"
            return
        if isinstance(node, exp.Func):
            name = (node.sql_name() or "").upper()
            if name in DENY_FUNCTIONS:
                yield f"包含禁止的函数: {name}"
                return
        if isinstance(node, exp.Anonymous):
            name = (node.name or "").upper()
            if name in DENY_FUNCTIONS:
                yield f"包含禁止的函数: {name}"
                return


def _collect_tables(stmt: exp.Expression) -> tuple[set[str], set[str]]:
    """返回 (物理表集合, CTE 别名集合)；CTE 别名不是物理表，不参与白名单校验。"""
    biz_db = (get_config().nl2sql.biz_database or "").lower()
    tables: set[str] = set()
    cte_names: set[str] = set()
    for cte in stmt.find_all(exp.CTE):
        alias = cte.alias
        if alias:
            cte_names.add(alias.lower())
    for node in stmt.find_all(exp.Table):
        name = node.name
        if name.lower() in cte_names:
            continue
        db = node.db
        if db and db.lower() != biz_db:
            tables.add(f"{db}.{name}".lower())
        else:
            tables.add(name.lower())
    return tables, cte_names


def _enforce_limit(stmt: exp.Expression, max_limit: int) -> exp.Expression:
    """对 SELECT / UNION 统一强制 LIMIT（缺失补默认值，超上限改写）。"""
    limit_expr = stmt.args.get("limit")
    if limit_expr is None:
        stmt.set("limit", exp.Limit(expression=exp.Literal.number(max_limit)))
        return stmt
    lval = limit_expr.expression
    num = lval.name if isinstance(lval, exp.Literal) and lval.is_int else None
    if num is None or int(num) > max_limit:
        stmt.set("limit", exp.Limit(expression=exp.Literal.number(max_limit)))
    return stmt


def _validate_shape(sql: str) -> GuardResult | None:
    """规范化后的二次校验：重新解析，确认仍是单条只读查询且无锁。"""
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="mysql") if s is not None]
    except sqlglot.errors.ParseError as e:
        return GuardResult(ok=False, reject_reason=f"SQL 规范化后解析失败: {e}")
    if len(statements) != 1 or not isinstance(statements[0], _query_types()):
        return GuardResult(ok=False, reject_reason="规范化后的语句不是只读 SELECT 查询")
    if statements[0].args.get("locks"):
        return GuardResult(ok=False, reject_reason="禁止 FOR UPDATE / FOR SHARE 锁读")
    return None


def check(
    sql: str,
    allowed_tables: set[str] | list[str],
    max_limit: int | None = None,
) -> GuardResult:
    """校验一条 SQL：通过返回 ok + 规范化 SQL + 涉及表；按白名单表做越权约束。"""
    cfg = get_config().nl2sql
    max_limit = max_limit or cfg.max_rows
    allowed = {t.lower() for t in allowed_tables}
    raw = (sql or "").strip().rstrip(";").strip()
    if not raw:
        return GuardResult(ok=False, reject_reason="SQL 为空")

    structural = _strip_literals(raw)
    if RE_MULTI_SEMI.search(structural):
        return GuardResult(ok=False, reject_reason="检测到多语句")
    if RE_VERSION_COMMENT.search(structural):
        return GuardResult(ok=False, reject_reason="包含版本注释(疑似绕过)")

    try:
        statements = sqlglot.parse(raw, dialect="mysql")
    except sqlglot.errors.ParseError as e:
        return GuardResult(ok=False, reject_reason=f"SQL 语法解析失败: {e}")
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return GuardResult(ok=False, reject_reason="仅允许单条语句")
    stmt = statements[0]
    if isinstance(stmt, exp.Subquery):
        stmt = stmt.this
    if not isinstance(stmt, _query_types()):
        return GuardResult(ok=False, reject_reason="仅允许 SELECT 查询")

    if RE_DANGEROUS_TEXT.search(_mask_literals(stmt)):
        return GuardResult(ok=False, reject_reason="包含危险关键字(系统库/系统变量/文件导出)")
    for reason in _iter_deny_nodes(stmt):
        return GuardResult(ok=False, reject_reason=reason)
    if stmt.args.get("locks"):
        return GuardResult(ok=False, reject_reason="禁止 FOR UPDATE / FOR SHARE 锁读")

    tables, _ = _collect_tables(stmt)
    unknown = tables - allowed
    if unknown:
        return GuardResult(
            ok=False,
            reject_reason=f"表不在白名单内: {sorted(unknown)}(可用表: {sorted(allowed)})",
        )

    stmt = _enforce_limit(stmt, max_limit)
    try:
        normalized = stmt.sql(dialect="mysql", comments=False)
    except Exception as e:
        return GuardResult(ok=False, reject_reason=f"SQL 规范化失败: {e}")

    shape_err = _validate_shape(normalized)
    if shape_err is not None:
        return shape_err

    return GuardResult(ok=True, normalized_sql=normalized, tables=sorted(tables))