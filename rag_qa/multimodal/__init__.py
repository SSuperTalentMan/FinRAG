#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/multimodal — 多模态分级文档解析层（融合 DocAudit parsing）。

页级路由：数字页 PyMuPDF 直提（零成本）→ 扫描页 RapidOCR（CPU）→ 低置信页 qwen-vl 兜底。
"""