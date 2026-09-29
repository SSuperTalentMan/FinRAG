#!/usr/bin/env python
"""
tests/test_enterprise_enhancements.py — 企业化增强项回归测试
覆盖：多策略检索路由（规则预筛 + 降级）、知识库属主权限、RAGAS 实验对比、
LLM 备用模型降级链、Milvus schema 漂移保护、会话消息原子追加与会话索引。
"""
import json
import time
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════
# 1. 多策略检索：规则预筛
# ═══════════════════════════════════════════════════════════════
class TestStrategyRules:
    def test_comparison_query_routes_to_subquery(self):
        from rag_qa.core.strategy_selector import STRATEGY_SUBQUERY, select_strategy_by_rules
        assert select_strategy_by_rules("存款保险和商业保险有什么区别") == STRATEGY_SUBQUERY
        assert select_strategy_by_rules("对比 Milvus 和 Elasticsearch 的优缺点") == STRATEGY_SUBQUERY

    def test_long_abstract_query_routes_to_hyde(self):
        from rag_qa.core.strategy_selector import STRATEGY_HYDE, select_strategy_by_rules
        assert select_strategy_by_rules("数字人民币的推广对商业银行的影响有哪些") == STRATEGY_HYDE

    def test_scenario_query_routes_to_backtrack(self):
        from rag_qa.core.strategy_selector import STRATEGY_BACKTRACK, select_strategy_by_rules
        assert select_strategy_by_rules("我有一个小型贸易公司，想申请银行贷款，应该怎么办理手续") == STRATEGY_BACKTRACK

    def test_plain_and_short_queries_stay_direct(self):
        from rag_qa.core.strategy_selector import STRATEGY_DIRECT, select_strategy_by_rules
        assert select_strategy_by_rules("银行贷款利率是多少") == STRATEGY_DIRECT
        assert select_strategy_by_rules("有哪些") == STRATEGY_DIRECT  # 短查询不路由
        assert select_strategy_by_rules("") == STRATEGY_DIRECT

    def test_select_strategy_entry_no_llm_when_rules_hit(self):
        """规则命中时不调用 LLM 选择器。"""
        from rag_qa.core import strategy_selector as ss
        with patch.object(ss, "StrategySelector") as mock_sel:
            result = ss.select_strategy("A和B的区别是什么", llm_fallback=True)
        assert result == ss.STRATEGY_SUBQUERY
        mock_sel.assert_not_called()

    def test_select_strategy_entry_llm_fallback_only_for_long_queries(self):
        """规则未命中 + 开启兜底 + 长查询 → 调 LLM；短查询不调。"""
        from rag_qa.core import strategy_selector as ss
        long_plain = "请介绍一下商业银行资本管理办法的主要内容和适用范围是怎样的呢"  # 无信号词
        with patch.object(ss, "StrategySelector") as mock_sel:
            mock_sel.return_value.select_strategy.return_value = ss.STRATEGY_DIRECT
            ss.select_strategy(long_plain, llm_fallback=True)
            mock_sel.return_value.select_strategy.assert_called_once()
            mock_sel.reset_mock()
            ss.select_strategy("银行贷款利率", llm_fallback=True)
            mock_sel.assert_not_called()

    def test_multi_strategy_disabled_forces_direct(self):
        """关闭 multi_strategy_enabled 时恒为直接检索，不触发任何策略调用。"""
        from routers.chat import STRATEGY_DIRECT, _build_context
        intent = MagicMock(domain="banking")
        mock_cfg = MagicMock()
        mock_cfg.retrieval.multi_strategy_enabled = False
        mock_cfg.retrieval.strategy_llm_fallback = False
        mock_cfg.retrieval.similarity_threshold = 0.0
        with (
            patch("routers.chat.get_config", return_value=mock_cfg),
            patch("routers.chat.record_strategy") as m_rec,
            patch("routers.chat._hybrid_recall", return_value=[
                {"question": "Q1", "answer": "A1", "category": "banking", "score": 0.9, "source": "bm25"},
            ]),
            patch("routers.chat.rerank", side_effect=lambda q, c, top_k: c[:top_k]),
        ):
            ctx, sources = _build_context("银行贷款利率是多少", intent)
        assert m_rec.call_args[0][0] == STRATEGY_DIRECT
        assert "A1" in ctx


class TestBuildContextStrategies:
    """_build_context 策略派发：改写类策略应调用 LLM 且失败降级直接召回。"""

    def _mock_cfg(self):
        mock_cfg = MagicMock()
        mock_cfg.retrieval.multi_strategy_enabled = True
        mock_cfg.retrieval.strategy_llm_fallback = False
        mock_cfg.retrieval.similarity_threshold = 0.0
        return mock_cfg

    def _candidate(self):
        return [{"question": "Q1", "answer": "A1", "category": "banking", "score": 0.9, "source": "bm25"}]

    def test_subquery_strategy_splits_and_merges(self):
        from routers.chat import _build_context
        intent = MagicMock(domain="banking")
        with (
            patch("routers.chat.get_config", return_value=self._mock_cfg()),
            patch("routers.chat.chat_completion", return_value="子问题一\n子问题二"),
            patch("routers.chat._hybrid_recall", return_value=self._candidate()) as m_recall,
            patch("routers.chat.rerank", side_effect=lambda q, c, top_k: c[:top_k]),
        ):
            ctx, sources = _build_context("存款准备金率和再贴现率的区别有哪些", intent)
        # 每个子查询召回一次，且精排用原始问题
        assert m_recall.call_count == 2
        assert "A1" in ctx

    def test_hyde_strategy_reranks_with_original_question(self):
        from routers.chat import _build_context
        intent = MagicMock(domain="banking")
        with (
            patch("routers.chat.get_config", return_value=self._mock_cfg()),
            patch("routers.chat.chat_completion", return_value="这是假设性答案文本"),
            patch("routers.chat._hybrid_recall", return_value=self._candidate()) as m_recall,
            patch("routers.chat.rerank", return_value=[]) as m_rerank,
        ):
            _build_context("数字人民币的推广对货币政策传导的影响有哪些", intent)
        # 召回用改写后的假设答案，精排用用户原始问题
        assert m_recall.call_args[0][0] == "这是假设性答案文本"
        assert m_rerank.call_args[0][0] == "数字人民币的推广对货币政策传导的影响有哪些"

    def test_rewrite_failure_falls_back_to_direct_recall(self):
        """LLM 改写失败时降级为原始查询直接召回（可用性优先）。"""
        from routers.chat import _build_context
        intent = MagicMock(domain="banking")
        with (
            patch("routers.chat.get_config", return_value=self._mock_cfg()),
            patch("routers.chat.chat_completion", side_effect=RuntimeError("llm down")),
            patch("routers.chat._hybrid_recall", return_value=self._candidate()) as m_recall,
            patch("routers.chat.rerank", side_effect=lambda q, c, top_k: c[:top_k]),
        ):
            ctx, sources = _build_context("存款准备金率和再贴现率的区别有哪些", intent)
        assert m_recall.call_count == 1
        assert m_recall.call_args[0][0] == "存款准备金率和再贴现率的区别有哪些"


# ═══════════════════════════════════════════════════════════════
# 2. 知识库属主权限
# ═══════════════════════════════════════════════════════════════
class TestCanManageKb:
    def test_admin_manages_everything(self):
        from services.authz import can_manage_kb
        admin = {"user_id": 2, "role": "admin"}
        assert can_manage_kb(admin, {"is_builtin": 1, "owner_id": None})
        assert can_manage_kb(admin, {"is_builtin": 0, "owner_id": 7})
        assert can_manage_kb(admin, None) is False

    def test_builtin_readonly_for_users(self):
        from services.authz import can_manage_kb
        user = {"user_id": 7, "role": "user"}
        assert can_manage_kb(user, {"is_builtin": 1, "owner_id": None}) is False

    def test_owner_can_manage_custom_kb(self):
        from services.authz import can_manage_kb
        user = {"user_id": 7, "role": "user"}
        assert can_manage_kb(user, {"is_builtin": 0, "owner_id": 7}) is True
        assert can_manage_kb(user, {"is_builtin": 0, "owner_id": 8}) is False
        assert can_manage_kb(user, {"is_builtin": 0, "owner_id": None}) is False


# ═══════════════════════════════════════════════════════════════
# 3. RAGAS 实验对比
# ═══════════════════════════════════════════════════════════════
class TestCompareExperiments:
    def _write_exp(self, path, name, faith, recall, samples=10):
        data = {
            "summary": {"faithfulness": faith, "llm_context_recall": recall},
            "num_samples": samples,
            "evaluated_at": int(time.time()),
        }
        (path / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_compare_two_experiments(self, tmp_path):
        from services import evaluation
        self._write_exp(tmp_path, "eval_111.json", 0.80, 0.70)
        self._write_exp(tmp_path, "eval_222.json", 0.86, 0.72)
        with patch.object(evaluation, "EXPERIMENTS_DIR", tmp_path):
            result = evaluation.compare_experiments(["eval_111.json", "eval_222.json"])
        assert "error" not in result
        assert result["metrics"]["faithfulness"]["best"] == "eval_222.json"
        assert result["metrics"]["faithfulness"]["delta"] == 0.06

    def test_compare_requires_two_files(self, tmp_path):
        from services import evaluation
        with patch.object(evaluation, "EXPERIMENTS_DIR", tmp_path):
            assert "error" in evaluation.compare_experiments(["only.json"])
            assert "error" in evaluation.compare_experiments([])

    def test_compare_missing_file_errors(self, tmp_path):
        from services import evaluation
        self._write_exp(tmp_path, "eval_111.json", 0.8, 0.7)
        with patch.object(evaluation, "EXPERIMENTS_DIR", tmp_path):
            result = evaluation.compare_experiments(["eval_111.json", "ghost.json"])
        assert "error" in result


# ═══════════════════════════════════════════════════════════════
# 4. LLM 备用模型降级链
# ═══════════════════════════════════════════════════════════════
class TestLlmFallbackChain:
    def _mock_cfg(self, fallback):
        cfg = MagicMock()
        cfg.llm.model = "primary-model"
        cfg.llm.fallback_model = fallback
        cfg.llm.temperature = 0.7
        cfg.llm.max_tokens = 128
        return cfg

    def test_fallback_model_used_on_primary_failure(self):
        from services import llm as llm_mod
        mock_client = MagicMock()
        primary_resp = MagicMock(); primary_resp.choices = [MagicMock()]; primary_resp.choices[0].message.content = "fallback-answer"
        primary_resp.usage = None
        mock_client.chat.completions.create.side_effect = [RuntimeError("rate limited"), primary_resp]
        with (
            patch.object(llm_mod, "get_config", return_value=self._mock_cfg("backup-model")),
            patch.object(llm_mod, "get_llm_client", return_value=mock_client),
        ):
            answer = llm_mod.chat_completion([{"role": "user", "content": "hi"}])
        assert answer == "fallback-answer"
        assert mock_client.chat.completions.create.call_count == 2
        assert mock_client.chat.completions.create.call_args_list[1].kwargs["model"] == "backup-model"

    def test_no_fallback_raises_original_error(self):
        from services import llm as llm_mod
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("boom")
        with (
            patch.object(llm_mod, "get_config", return_value=self._mock_cfg("")),
            patch.object(llm_mod, "get_llm_client", return_value=mock_client),pytest.raises(RuntimeError)
        ):
            llm_mod.chat_completion([{"role": "user", "content": "hi"}])
        assert mock_client.chat.completions.create.call_count == 1


# ═══════════════════════════════════════════════════════════════
# 5. Milvus schema 漂移保护
# ═══════════════════════════════════════════════════════════════
class TestMilvusSchemaGuard:
    def _fake_client(self, fields):
        client = MagicMock()
        client.has_collection.return_value = True
        client.describe_collection.return_value = {"fields": [{"name": f} for f in fields]}
        return client

    def test_schema_mismatch_raises_instead_of_drop(self, monkeypatch):
        import db.milvus as m
        monkeypatch.setattr(m, "_collection_ready", False)
        monkeypatch.delenv("MILVUS_AUTO_RECREATE", raising=False)
        client = self._fake_client(["id", "dense_vector"])  # 缺 sparse_vector
        with pytest.raises(RuntimeError) as exc:
            m.ensure_collection(client)
        assert "schema 不匹配" in str(exc.value)
        client.drop_collection.assert_not_called()

    def test_auto_recreate_escape_hatch(self, monkeypatch):
        import db.milvus as m
        monkeypatch.setattr(m, "_collection_ready", False)
        monkeypatch.setenv("MILVUS_AUTO_RECREATE", "1")
        client = self._fake_client(["id", "dense_vector"])
        m.ensure_collection(client)
        client.drop_collection.assert_called_once()
        client.create_collection.assert_called_once()

    def test_valid_schema_no_drop(self, monkeypatch):
        import db.milvus as m
        monkeypatch.setattr(m, "_collection_ready", False)
        client = self._fake_client(["id", "dense_vector", "sparse_vector"])
        m.ensure_collection(client)
        client.drop_collection.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# 6. 会话消息原子追加（降级路径）+ 会话索引回退
# ═══════════════════════════════════════════════════════════════
class _FakePipeline:
    def __init__(self, store):
        self._store = store
        self._keys = []

    def get(self, key):
        self._keys.append(key)

    def execute(self):
        return [self._store.get(k) for k in self._keys]


class _FakeRedis:
    """最小 Redis 桩：支持 get/setex/zadd/zrem/zrevrange/expire/scan_iter/pipeline。"""

    def __init__(self, store=None):
        self.store = store if store is not None else {}
        self.zadds = []

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)

    def register_script(self, lua):
        raise RuntimeError("test stub: lua not available")

    def zadd(self, key, mapping):
        self.zadds.append((key, dict(mapping)))
        cur = self.store.setdefault(key, {})
        cur.update(mapping)

    def zrem(self, key, *members):
        cur = self.store.get(key) or {}
        for m in members:
            cur.pop(m, None)

    def zrevrange(self, key, start, end):
        cur = self.store.get(key) or {}
        items = sorted(cur.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in items]

    def expire(self, key, ttl):
        pass

    def scan_iter(self, pattern):
        for k in list(self.store.keys()):
            if k.startswith(pattern.rstrip("*")):
                yield k

    def pipeline(self):
        return _FakePipeline(self.store)


class TestSessionAtomicAppend:
    def test_append_fallback_path_appends_and_trims(self):
        """Lua 不可用时降级读改写：消息追加、超限裁剪、user_id 绑定。"""
        import db.redis as rmod
        fake = _FakeRedis()
        with patch.object(rmod, "get_redis", return_value=fake):
            rmod.append_session_message(
                "s1", {"role": "user", "content": "m1", "ts": 1}, user_id=7, max_messages=2
            )
            rmod.append_session_message(
                "s1", {"role": "assistant", "content": "m2", "ts": 2}, user_id=7, max_messages=2
            )
            rmod.append_session_message(
                "s1", {"role": "user", "content": "m3", "ts": 3}, user_id=7, max_messages=2
            )
        data = json.loads(fake.store["finrag:session:s1"])
        assert data["user_id"] == 7
        assert [m["content"] for m in data["messages"]] == ["m2", "m3"]

    def test_append_refreshes_session_index(self):
        import db.redis as rmod
        fake = _FakeRedis()
        with patch.object(rmod, "get_redis", return_value=fake):
            rmod.append_session_message(
                "s1", {"role": "user", "content": "m1", "ts": 42}, user_id=7
            )
        assert fake.store["finrag:user_sessions:7"]["s1"] == 42


class TestGetUserSessions:
    def test_scan_fallback_filters_by_owner_and_rebuilds_index(self):
        """索引为空时回退 scan：只返回该用户的会话，并顺带重建索引。"""
        import db.redis as rmod
        store = {
            "finrag:session:a": json.dumps({"user_id": 7, "created_at": 1, "messages": [{"role": "user", "content": "q", "ts": 10}]}),
            "finrag:session:b": json.dumps({"user_id": 8, "created_at": 2, "messages": [{"role": "user", "content": "q", "ts": 20}]}),
        }
        fake = _FakeRedis(store)
        with patch.object(rmod, "get_redis", return_value=fake):
            result = rmod.get_user_sessions(7)
        assert [sid for sid, _ in result] == ["a"]
        # 索引已重建
        assert fake.store["finrag:user_sessions:7"]["a"] == 10

    def test_index_path_lazy_prunes_expired(self):
        """索引路径：已过期的会话条目被惰性剔除。"""
        import db.redis as rmod
        store = {
            "finrag:user_sessions:7": {"alive": 20, "expired": 10},
            "finrag:session:alive": json.dumps({"user_id": 7, "created_at": 1, "messages": []}),
        }
        fake = _FakeRedis(store)
        with patch.object(rmod, "get_redis", return_value=fake):
            result = rmod.get_user_sessions(7)
        assert [sid for sid, _ in result] == ["alive"]
        assert "expired" not in fake.store["finrag:user_sessions:7"]
