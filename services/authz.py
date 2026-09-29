#!/usr/bin/env python
"""
services/authz.py — 资源访问控制（属主模型）

规则（最小权限 + 管理员全覆盖）：
- 管理员：可管理全部资源；
- 内置知识库（is_builtin=1）：属主为系统（owner_id=NULL），普通用户只读；
- 自定义知识库：仅创建者（owner_id）与管理员可写。

知识库的写操作（改名/删除/上传文档/删除文档/停止解析）统一走本模块校验，
读操作（列表/检索）对所有登录用户开放。
"""



def can_manage_kb(user: dict, kb: dict | None) -> bool:
    """判断用户是否有权对知识库执行写操作（含其下文档的管理）。

    user: get_current_user 返回的 {"user_id", "username", "role"}
    kb:   get_kb 返回的知识库行（含 is_builtin / owner_id），可为 None（不存在时返回 False）
    """
    if not user or not kb:
        return False
    if user.get("role") == "admin":
        return True
    if kb.get("is_builtin"):
        return False
    owner_id = kb.get("owner_id")
    return owner_id is not None and owner_id == user.get("user_id")
