# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

尝尝咸淡 RAG 系统 — 基于 LangChain + FAISS + DeepSeek 的中文烹饪食谱检索增强生成（RAG）系统。支持从上百道中文食谱 Markdown 文档中智能检索，并通过 LLM 生成分步骤的烹饪指导回答。

## 环境与运行

```bash
# 安装依赖
pip install -r code/requirements.txt

# 必须设置的环境变量（也可在 code/.env 中配置，项目使用 python-dotenv 自动加载）
export DEEPSEEK_API_KEY="your-api-key"

# 方式一：命令行交互
python code/main.py

# 方式二：Web 界面（推荐）
python code/api_server.py
# 然后在浏览器打开 frontend/index.html
```

所有命令从项目根目录执行。`code/main.py` 会自动将 `code/` 作为基准目录，解析配置文件中的相对路径。没有测试套件。

## 查询处理全流程

`RecipeRAGSystem.ask_question()` 中的完整查询流水线：

```text
用户问题
  → intent_guard (意图守卫：调用 LLM 判断是否与烹饪相关，拦截闲聊)
  → query_router (路由分类：recommend 推荐 / cook 具体做法)
  → query_rewrite (模糊查询自动补全为精确搜索词；recommend 类跳过此步)
  → _extract_filters_from_query (从问题文本中匹配分类/难度关键词作为元数据过滤条件)
  → hybrid_search (FAISS 向量 + BM25 关键词，各取 top 5，RRF 重排 + 菜名匹配加权)
  → metadata_filtered_search (如有过滤条件则先扩大检索再过滤；无结果时回退到无过滤)
  → get_parent_documents (从检索到的子块反查完整父文档，按匹配次数去重排序)
  → generate (cook → 分步骤指导; recommend → 列表式回答; 均可选 SSE 流式)
```

## 核心架构

四个 RAG 模块 + 一个工具模块，由 `code/main.py` 中的 `RecipeRAGSystem` 类编排：

### 1. 数据准备 — `code/rag_modules/data_preparation.py`

`DataPreparationModule` 负责：
- 递归读取 `data/cook/dishes/` 下所有 `.md` 食谱文件，创建父文档（完整食谱）
- 从文件夹路径推断分类（如 `meat_dish` → 荤菜），从 `★` 符号推断难度
- 使用 `MarkdownHeaderTextSplitter` 按 `#`/`##`/`###` 标题层级分割为子块（chunk），建立父子文档映射（`parent_child_map`）
- 分类和难度的标签集定义在 `CATEGORY_MAPPING` 和 `DIFFICULTY_LABELS` 类常量中，供其他模块引用
- 父子文档通过 `parent_id`（MD5 of 相对路径）关联；`get_parent_documents()` 使用 source 路径优先匹配、parent_id 回退的策略

### 2. 索引构建 — `code/rag_modules/index_construction.py`

`IndexConstructionModule` 负责：
- 使用 `BAAI/bge-small-zh-v1.5` 中文嵌入模型（CPU 运行）
- 嵌入模型加载优先级：直接路径 → `code/models/{model_name}/` → HF 缓存目录（`~/.cache/huggingface/hub/`），启动时设置 `HF_HUB_OFFLINE=1` 避免联网
- 构建 FAISS 向量索引，支持保存到磁盘和从磁盘加载
- 已构建的索引存储在 `code/vector_index/`（`index.faiss` + `index.pkl`）
- 加载索引时使用 `allow_dangerous_deserialization=True`（FAISS 依赖 pickle）

### 3. 检索优化 — `code/rag_modules/retrieval_optimization.py`

`RetrievalOptimizationModule` 负责：
- **混合检索**：FAISS 向量检索 + BM25 关键词检索，各取 top 5
- **RRF 重排**（Reciprocal Rank Fusion）：合并两种检索结果，按 `1/(k+rank+1)` 公式计分，同时菜名命中查询核心词时额外加权
- **元数据过滤**：先扩大检索（top_k × 3）再按 category/difficulty 过滤
- BM25 检索器在 `setup_retrievers()` 时从 chunks 构建，使用 `rank_bm25` 库

### 4. 生成集成 — `code/rag_modules/generation_integration.py`

`GenerationIntegrationModule` 负责：
- 调用 DeepSeek Chat API（通过 `langchain_deepseek.ChatDeepSeek`，始终启用 streaming）
- **意图守卫** (`intent_guard`)：判断问题是否与烹饪相关，返回 YES/NO
- **查询路由** (`query_router`)：将用户问题分类为 `recommend`（推荐）或 `cook`（做法）
- **查询重写** (`query_rewrite`)：对模糊查询自动补全为更精确的搜索词（具体菜名查询保持原样）
- 三种回答模板：`generate_basic_answer`、`generate_step_by_step_answer`（含菜品介绍/食材/步骤/技巧）、`generate_list_answer`
- 每种模板均有对应的流式版本（`_stream` 后缀）
- `_build_context()` 限制上下文最长 2000 字符，按文档截断

### 5. 农历工具 — `code/lunar_utils.py`

纯 Python 实现，零外部依赖，用于「每日推荐」功能的日期上下文：
- 公历 → 农历日期转换（2020–2035 年数据，含闰月）
- 24 节气查询（近似日期 ±1 天精度）
- 干支纪年、生肖
- 季节判断（基于节气分界：春分/夏至/秋分/冬至）
- `format_date_context()` 返回完整日期上下文字典

## 每日推荐功能

`RecipeRAGSystem.get_daily_recommendation()` 实现，流程为：

1. **缓存检查**：同一天返回相同结果
2. **日期上下文**：调用 `format_date_context()` 获取农历、节气、季节
3. **去重**：加载推荐历史，排除最近 7 天推荐过的菜品
4. **规则引擎初筛**：`_score_dishes_for_recommendation()` 根据季节加权 + 工作日/周末偏好（工作日加权简单菜，周末加权复杂菜）+ 随机扰动，取 top 15
5. **LLM 精选**：`_llm_pick_dish()` 让 LLM 从候选 15 道菜中选出 1 道并写推荐理由（带 JSON 解析和 fallback）
6. **保存历史**：写入 `code/recommendation_history.json`，保留最近 30 天

对应的 API 端点为 `GET /api/daily-recommendation`。

## Web 前端 & API

### API 服务 — `code/api_server.py`

FastAPI 服务，封装 `RecipeRAGSystem` 为 REST API（默认端口 8899）：

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/health` | GET | 健康检查，返回系统启动状态和错误详情 |
| `/api/ask` | POST | 问答接口，`stream:true` 时 SSE 流式返回 |
| `/api/categories` | GET | 分类和难度标签列表 |
| `/api/recipes` | GET | 按分类/难度筛选食谱 |
| `/api/stats` | GET | 知识库统计信息 |
| `/api/daily-recommendation` | GET | 每日菜品推荐（含农历/节气上下文 + 推荐理由） |

启动时在 `@app.on_event("startup")` 中初始化 RAG 系统，确保 `vector_index/` 目录存在。所有端点通过 `_check_ready()` 守卫，启动失败时返回 503。

### 前端页面 — `frontend/index.html`

纯 HTML/CSS/JS 单页应用，零构建依赖。设计主题为"灶台书香"——暖陶土色调、仿古纸纹理、中文衬线字体。支持：
- 自然语言搜索 + 分类/难度 chip 筛选
- SSE 流式逐字输出（打字机效果）
- 三种结果模式自动切换：菜品卡片列表 / 分步骤指导 / 一般问答
- 食谱详情弹窗
- 每日推荐展示
- 底部知识库统计

## 配置

`code/config.py` 中的 `RAGConfig` dataclass 包含所有可调参数：
- `data_path`: 食谱数据路径（默认 `../data/cook/dishes`，在 `main.py` 中自动基于 `code/` 目录解析为绝对路径）
- `index_save_path`: 索引存储路径（默认 `./vector_index`，同样基于 `code/` 解析）
- `embedding_model`: 默认 `BAAI/bge-small-zh-v1.5`
- `llm_model`: 默认 `deepseek-v4-flash`
- `top_k`: 检索返回数量，默认 3
- `temperature`: 生成温度，默认 0.1
- `max_tokens`: 最大生成 token 数，默认 2048

## 数据组织

- 食谱以 Markdown 文件存储在 `data/cook/dishes/` 下
- 按分类分文件夹：`meat_dish`/荤菜、`vegetable_dish`/素菜、`soup`/汤品、`dessert`/甜品、`staple`/主食、`breakfast`/早餐、`aquatic`/水产、`condiment`/调料、`drink`/饮品、`semi-finished`/半成品
- 每道菜可包含图片（jpg/png/webp），位于菜品同名子文件夹中
- 食谱 Markdown 结构：`# 菜名` → `## 必备原料和工具` → `## 计算` → `## 操作` → `## 附加内容`，难度用 ★ 数量标记

## 重要注意事项

- **嵌入模型离线优先**：`IndexConstructionModule._resolve_local_model_path()` 会先在 `code/models/` 和 HF 缓存中查找，避免每次启动联网。首次运行需确保模型已缓存到本地。
- **FAISS 索引依赖 pickle**：加载时需 `allow_dangerous_deserialization=True`，仅加载可信来源的索引文件。
- **BM25 检索器依赖 chunks**：`RetrievalOptimizationModule` 初始化时必须传入完整的 chunks 列表用于构建 BM25 索引。
- **索引目录**：FAISS 保存索引前需要父目录已存在，`api_server.py` 在启动时自动创建 `code/vector_index/`。
- **DeepSeek API**：`ChatDeepSeek` 始终以 `streaming=True` 初始化，非流式回答通过拼接流式 chunk 实现。
