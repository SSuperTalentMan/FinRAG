#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_qa/edu_text_spliter/edu_chinese_recursive_text_splitter.py

中文递归文本切分器，基于 LangChain RecursiveCharacterTextSplitter，
针对中文语境优化分隔符优先级（句号 > 换行 > 分号 > 逗号）。
"""

import re
from typing import List, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _split_text_with_regex_from_end(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """
    中文递归文本切分器。

    分隔符优先级（从高到低）：
      1. 双换行（段落边界）
      2. 单换行
      3. 中文句末标点（。！？）
      4. 英文句末标点（. ! ?）
      5. 中文分号（；）
      6. 中文逗号（，）
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            r"\.\s*|\!\s*|\?\s*",
            r"；|;\s*",
            r"，|,\s*",
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        _good_splits = []
        _separator_str = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged = self._merge_splits(_good_splits, _separator_str)
                    final_chunks.extend(merged)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other = self._split_text(s, new_separators)
                    final_chunks.extend(other)
        if _good_splits:
            merged = self._merge_splits(_good_splits, _separator_str)
            final_chunks.extend(merged)

        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()]
