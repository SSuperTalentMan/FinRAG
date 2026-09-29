#!/usr/bin/env python
"""
tests/test_security_hardening.py — 本轮安全与健壮性加固的回归测试
覆盖：
  1. get_current_user 回查 DB：用户被删除/禁用/降级后，旧 Token 立即失效或降权
  2. verify_password 不再吞掉禁用状态（登录接口可区分 401/403）
  3. 上传文档 domain 解析：回退知识库 domain + 格式白名单校验
  4. LLM 配额检查：超限抛 429
  5. 会话消息长度上限
  6. BM25 索引原子重建：构建失败保留旧索引，快照内部一致
"""
import time
from unittest.mock import patch

import bcrypt
import pytest
from fastapi import HTTPException
from pydantic import ValidationError


# ═══════════════════════════════════════════════════════════════
# 1. get_current_user：Token 有效 ≠ 账号可用
# ═══════════════════════════════════════════════════════════════
class TestGetCurrentUserDbValidation:
    def _make_token(self, uid=42, username="alice", role="user"):
        from routers.auth import _create_token
        return _create_token(uid, username, role)

    def _get_credentials(self, token):
        from fastapi.security.http import HTTPAuthorizationCredentials
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    def test_active_user_returns_db_role(self):
        """正常用户：返回 DB 中的角色与用户信息。"""
        from routers.auth import get_current_user
        user_row = {"id": 42, "username": "alice", "role": "user", "status": "active"}
        with patch("routers.auth.get_user_by_id", return_value=user_row):
            info = get_current_user(self._get_credentials(self._make_token()))
        assert info == {"username": "alice", "user_id": 42, "role": "user"}

    def test_demoted_admin_gets_db_role(self):
        """管理员被降级后，即使 Token 里仍是 admin 角色，也应拿到 DB 的 user 角色。"""
        from routers.auth import get_current_user
        # Token 伪造为 admin，但 DB 中已降级为 user
        token = self._make_token(uid=42, username="alice", role="admin")
        user_row = {"id": 42, "username": "alice", "role": "user", "status": "active"}
        with patch("routers.auth.get_user_by_id", return_value=user_row):
            info = get_current_user(self._get_credentials(token))
        assert info["role"] == "user"

    def test_deleted_user_rejected(self):
        """用户被删除后，旧 Token 立即失效（401），而非等 24h 过期。"""
        from routers.auth import get_current_user
        with patch("routers.auth.get_user_by_id", return_value=None), pytest.raises(HTTPException) as exc:
            get_current_user(self._get_credentials(self._make_token()))
        assert exc.value.status_code == 401

    def test_disabled_user_rejected(self):
        """用户被禁用后，旧 Token 立即被拒（403）。"""
        from routers.auth import get_current_user
        user_row = {"id": 42, "username": "alice", "role": "user", "status": "disabled"}
        with patch("routers.auth.get_user_by_id", return_value=user_row), pytest.raises(HTTPException) as exc:
            get_current_user(self._get_credentials(self._make_token()))
        assert exc.value.status_code == 403

    def test_missing_credentials_rejected(self):
        from routers.auth import get_current_user
        with pytest.raises(HTTPException) as exc:
            get_current_user(None)
        assert exc.value.status_code == 401


# ═══════════════════════════════════════════════════════════════
# 2. verify_password：禁用用户也能通过密码验证（由登录接口给 403）
# ═══════════════════════════════════════════════════════════════
class TestVerifyPasswordDisabled:
    def _hash(self, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def test_disabled_user_with_correct_password_returns_user(self):
        """禁用用户密码正确时应返回用户信息，由登录接口区分 403（而非误导性的 401）。"""
        from db.mysql import verify_password
        user_row = {
            "id": 7, "username": "bob", "role": "user",
            "status": "disabled", "password_hash": self._hash("secret123"),
        }
        with patch("db.mysql.get_user_by_username", return_value=user_row):
            user = verify_password("bob", "secret123")
        assert user is not None
        assert user["status"] == "disabled"
        assert "password_hash" not in user

    def test_disabled_user_with_wrong_password_returns_none(self):
        from db.mysql import verify_password
        user_row = {
            "id": 7, "username": "bob", "role": "user",
            "status": "disabled", "password_hash": self._hash("secret123"),
        }
        with patch("db.mysql.get_user_by_username", return_value=user_row):
            assert verify_password("bob", "wrong") is None

    def test_unknown_user_returns_none(self):
        """用户不存在：走伪造哈希校验（防时序枚举），返回 None。"""
        from db.mysql import verify_password
        with patch("db.mysql.get_user_by_username", return_value=None):
            assert verify_password("ghost", "whatever") is None


# ═══════════════════════════════════════════════════════════════
# 3. 上传文档 domain 解析
# ═══════════════════════════════════════════════════════════════
class TestResolveUploadDomain:
    def test_provided_domain_wins(self):
        from routers.document import _resolve_upload_domain
        assert _resolve_upload_domain("banking", "fintech") == "banking"

    def test_fallback_to_kb_domain(self):
        """前端上传不传 domain 时回退知识库 domain（否则块落入 general，检索不可达）。"""
        from routers.document import _resolve_upload_domain
        assert _resolve_upload_domain(None, "banking") == "banking"

    def test_fallback_empty_kb_domain_becomes_general(self):
        from routers.document import _resolve_upload_domain
        assert _resolve_upload_domain(None, "") == "general"
        assert _resolve_upload_domain(None, None) == "general"

    def test_fallback_invalid_kb_domain_becomes_general(self):
        from routers.document import _resolve_upload_domain
        assert _resolve_upload_domain(None, "银行 领域!") == "general"

    def test_provided_invalid_domain_raises_400(self):
        from routers.document import _resolve_upload_domain
        with pytest.raises(HTTPException) as exc:
            _resolve_upload_domain('x" or 1==1 //', "banking")
        assert exc.value.status_code == 400
        with pytest.raises(HTTPException):
            _resolve_upload_domain("a" * 51, "")

    def test_provided_valid_domain_formats(self):
        from routers.document import _resolve_upload_domain
        assert _resolve_upload_domain("risk_management", "") == "risk_management"
        assert _resolve_upload_domain("kb01", "") == "kb01"


# ═══════════════════════════════════════════════════════════════
# 4. LLM 配额检查：超限必须抛 429（流式接口开流前检查）
# ═══════════════════════════════════════════════════════════════
class TestLlmQuota:
    def test_quota_exceeded_raises_429(self):
        from routers.chat import _check_llm_quota
        with (
            patch("routers.chat.check_rate_limit", return_value=(False, 101)),
            pytest.raises(HTTPException) as exc,
        ):
            _check_llm_quota(42)
        assert exc.value.status_code == 429

    def test_quota_ok_passes(self):
        from routers.chat import _check_llm_quota
        with patch("routers.chat.check_rate_limit", return_value=(True, 1)):
            _check_llm_quota(42)  # 不抛异常即通过


# ═══════════════════════════════════════════════════════════════
# 5. 会话消息长度上限
# ═══════════════════════════════════════════════════════════════
class TestSaveMessageRequestLimit:
    def test_content_within_limit_ok(self):
        from routers.conversation import SaveMessageRequest
        req = SaveMessageRequest(session_id="s1", role="user", content="x" * 16000)
        assert len(req.content) == 16000

    def test_content_over_limit_rejected(self):
        from routers.conversation import SaveMessageRequest
        with pytest.raises(ValidationError):
            SaveMessageRequest(session_id="s1", role="user", content="x" * 16001)


# ═══════════════════════════════════════════════════════════════
# 6. BM25 索引原子重建
# ═══════════════════════════════════════════════════════════════
class TestBM25AtomicRebuild:
    _ITEMS = [
        {"id": 1, "category": "banking", "question": "银行贷款年利率是多少",
         "answer": "按政策而定", "source": "mysql", "type": "事实性"},
        {"id": 2, "category": "fintech", "question": "数字人民币如何发行",
         "answer": "由央行发行", "source": "mysql", "type": "事实性"},
    ]

    def _new_retriever(self):
        from services.bm25 import BM25Retriever
        return BM25Retriever()

    def test_rebuild_builds_consistent_snapshot(self):
        retriever = self._new_retriever()
        with patch("services.bm25.get_all_qa_for_bm25", return_value=list(self._ITEMS)):
            retriever._rebuild()
        assert retriever._loaded is True
        assert len(retriever._docs) == 2
        assert retriever._bm25 is not None
        # 快照内部一致：bm25 与 docs 条数匹配，检索可用（search 按分数降序返回全部候选）
        hits = retriever.search("银行贷款")
        assert hits[0]["category"] == "banking"
        # 领域桶可用：指定 domain 时只返回该领域的文档
        hits = retriever.search("发行", domain="fintech")
        assert len(hits) == 1
        assert hits[0]["category"] == "fintech"

    def test_rebuild_failure_keeps_old_index(self):
        retriever = self._new_retriever()
        with patch("services.bm25.get_all_qa_for_bm25", return_value=list(self._ITEMS)):
            retriever._rebuild()
        old_docs = retriever._docs
        # 重建失败（DB 异常）：旧索引必须原样保留
        with patch("services.bm25.get_all_qa_for_bm25", side_effect=RuntimeError("db down")):
            retriever._rebuild()
        assert retriever._docs is old_docs
        assert retriever._bm25 is not None
        assert retriever.search("银行贷款")[0]["category"] == "banking"

    def test_legacy_attribute_setters_still_work(self):
        """兼容旧用法：直接给 _docs / _bm25 赋值（如测试 mock）仍能正常检索。"""
        from rank_bm25 import BM25Okapi

        from services.bm25 import tokenize
        retriever = self._new_retriever()
        retriever._docs = list(self._ITEMS)
        retriever._bm25 = BM25Okapi([tokenize(d["question"]) for d in self._ITEMS])
        retriever._loaded = True
        retriever._last_built = time.time()
        hits = retriever.search("数字人民币")
        # 旧属性赋值方式构建的索引可正常检索（候选中包含 fintech 文档即可，
        # 具体排序取决于 jieba 分词与 BM25 IDF，不做强断言）
        assert "fintech" in {h["category"] for h in hits}
