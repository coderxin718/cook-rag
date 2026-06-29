# -*- coding: utf-8 -*-
"""
尝尝咸淡 RAG 系统 — FastAPI 后端服务
"""
import os
import sys
import json
import uuid
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

# 确保可以导入同目录下的模块
sys.path.insert(0, str(Path(__file__).parent))

import redis.asyncio as aioredis
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import RecipeRAGSystem
from rag_modules.data_preparation import DataPreparationModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("api_server")

# ---------------------------------------------------------------------------
# Redis 会话管理器
# ---------------------------------------------------------------------------

class RedisSessionManager:
    """基于 Redis 的会话管理器。

    Key 结构:
      cookrag:session:{sid}   → Hash    (字段: created_at, last_active, message_count)
      cookrag:history:{sid}   → List    (每条是 JSON 消息, LPUSH 追加)
    TTL: 两个 key 均设置 EXPIRE，后续每次操作续期。
    """

    def __init__(self, redis_url: str, prefix: str = "cookrag", ttl: int = 86400):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix
        self.ttl = ttl
        self._ready = False

    async def _ensure_connection(self):
        """延迟验证连接（避免启动时 Redis 不可用导致崩溃）。"""
        if self._ready:
            return True
        try:
            await self.redis.ping()
            self._ready = True
            return True
        except Exception:
            return False

    def _session_key(self, sid: str) -> str:
        return f"{self.prefix}:session:{sid}"

    def _history_key(self, sid: str) -> str:
        return f"{self.prefix}:history:{sid}"

    async def create(self) -> str:
        """创建新会话，返回 session_id。"""
        sid = uuid.uuid4().hex[:8]
        now = datetime.now().isoformat()
        if await self._ensure_connection():
            pipe = self.redis.pipeline()
            pipe.hset(self._session_key(sid), mapping={
                "created_at": now, "last_active": now, "message_count": "0"
            })
            pipe.expire(self._session_key(sid), self.ttl)
            await pipe.execute()
        return sid

    async def exists(self, sid: str) -> bool:
        if not await self._ensure_connection():
            return False
        return await self.redis.exists(self._session_key(sid)) > 0

    async def get_history(self, sid: str, max_turns: int = 10) -> list:
        """获取会话的消息历史（时间升序：旧→新）。"""
        if not await self._ensure_connection():
            return []
        raw_list = await self.redis.lrange(self._history_key(sid), 0, -1)
        # 刷新 TTL
        pipe = self.redis.pipeline()
        pipe.expire(self._session_key(sid), self.ttl)
        pipe.expire(self._history_key(sid), self.ttl)
        await pipe.execute()
        # 倒序：Redis List LPUSH index 0 是最新，翻转为时间升序
        messages = [json.loads(item) for item in reversed(raw_list)]
        # 只返回最近 max_turns 轮（每轮 = user + assistant）
        if max_turns > 0 and len(messages) > max_turns * 2:
            messages = messages[-(max_turns * 2):]
        return messages

    async def append(self, sid: str, role: str, content: str):
        """追加一条消息到会话历史。"""
        if not await self._ensure_connection():
            return
        msg = json.dumps({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False)
        pipe = self.redis.pipeline()
        pipe.lpush(self._history_key(sid), msg)
        pipe.expire(self._history_key(sid), self.ttl)
        pipe.hset(self._session_key(sid), "last_active", datetime.now().isoformat())
        pipe.hincrby(self._session_key(sid), "message_count", 1)
        pipe.expire(self._session_key(sid), self.ttl)
        await pipe.execute()

    async def delete(self, sid: str):
        """删除会话及其历史。"""
        if not await self._ensure_connection():
            return
        pipe = self.redis.pipeline()
        pipe.delete(self._session_key(sid), self._history_key(sid))
        await pipe.execute()

    async def close(self):
        await self.redis.aclose()


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="尝尝咸淡 AI 智能食谱助手",
    description="基于 RAG 的中文烹饪食谱检索与生成系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
rag_system: Optional[RecipeRAGSystem] = None
session_manager: Optional[RedisSessionManager] = None
_startup_ok: bool = False
_startup_error: str = ""


@app.on_event("startup")
def startup():
    global rag_system, session_manager, _startup_ok, _startup_error
    # Windows 下强制 stdout 使用 UTF-8，避免 emoji 打印报错
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("\n--- 正在启动 RAG 系统...", flush=True)

    # 确保 vector_index 目录存在（FAISS C++ 底层需要父目录预先存在）
    _code_dir = Path(__file__).parent
    _index_dir = _code_dir / "vector_index"
    _index_dir.mkdir(parents=True, exist_ok=True)
    print(f"   [OK] 索引目录已就绪: {_index_dir}", flush=True)

    try:
        rag_system = RecipeRAGSystem()
        print("   [OK] RecipeRAGSystem 创建成功", flush=True)
        rag_system.initialize_system()
        print("   [OK] 模块初始化完成", flush=True)
        rag_system.build_knowledge_base()
        print("   [OK] 知识库构建完成", flush=True)

        # 初始化 Redis 会话管理器
        _redis_url = os.getenv("REDIS_URL", rag_system.config.redis_url)
        _ttl = rag_system.config.session_ttl_seconds
        _prefix = rag_system.config.redis_key_prefix
        try:
            session_manager = RedisSessionManager(
                redis_url=_redis_url,
                prefix=_prefix,
                ttl=_ttl,
            )
            print(f"   [OK] Redis 会话管理器已初始化 ({_redis_url})", flush=True)
        except Exception as e:
            logger.warning(f"Redis 连接失败，回退到无会话模式: {e}")
            print(f"   [WARN] Redis 不可用，多轮对话功能暂时不可用: {e}", flush=True)
            session_manager = None

        # 预热每日推荐缓存（首次需调 LLM，启动时算好，用户打开即秒出）
        try:
            print("   [..] 预热每日推荐缓存...", flush=True)
            rag_system.get_daily_recommendation()
            print("   [OK] 每日推荐缓存已预热", flush=True)
        except Exception as e:
            logger.warning(f"预热每日推荐失败（不影响主流程）: {e}")

        _startup_ok = True
        print("[OK] RAG 系统启动完成!\n", flush=True)
    except Exception as e:
        _startup_error = str(e)
        _startup_ok = False
        rag_system = None
        traceback.print_exc()
        print(f"\n[ERROR] RAG 系统启动失败: {e}\n", flush=True)
        logger.error(f"RAG 系统启动失败: {e}")


@app.on_event("shutdown")
async def shutdown():
    """关闭时清理 Redis 连接"""
    global session_manager
    if session_manager:
        try:
            await session_manager.close()
            logger.info("Redis 会话管理器已关闭")
        except Exception as e:
            logger.warning(f"关闭 Redis 连接时出错: {e}")


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    stream: bool = True
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------

def _check_ready():
    """检查系统是否就绪，返回 None 表示 OK，否则返回错误响应"""
    if not _startup_ok:
        return JSONResponse(
            status_code=503,
            content={"error": f"RAG 系统启动失败: {_startup_error or '未知错误'}"},
        )
    if rag_system is None:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG 系统尚未就绪，请稍后重试"},
        )
    return None


@app.get("/api/health")
async def health():
    """健康检查 — 返回系统状态和启动错误详情"""
    return {
        "status": "ok" if _startup_ok else "error",
        "startup_error": _startup_error or None,
        "rag_ready": rag_system is not None,
    }


@app.post("/api/session/new")
async def create_session():
    """创建新会话"""
    err = _check_ready()
    if err:
        # 系统未就绪时仍然允许创建本地会话
        sid = uuid.uuid4().hex[:8]
        return {"session_id": sid, "message": "新会话已创建（离线模式）"}

    if session_manager:
        sid = await session_manager.create()
    else:
        sid = uuid.uuid4().hex[:8]
    return {"session_id": sid, "message": "新会话已创建"}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话及其历史"""
    if session_manager:
        await session_manager.delete(session_id)
    return {"message": "会话已删除"}


@app.post("/api/ask")
async def ask_recipe(request: AskRequest):
    """智能问答接口 — 支持多轮对话（可选 session_id）和 SSE 流式"""
    err = _check_ready()
    if err:
        return err

    # ---- 会话处理 ----
    session_id = request.session_id
    chat_history = []

    if session_id and session_manager:
        # 确保会话存在，不存在则自动创建
        if not await session_manager.exists(session_id):
            session_id = await session_manager.create()
        chat_history = await session_manager.get_history(
            session_id, max_turns=rag_system.config.max_history_turns
        )
    elif session_id and not session_manager:
        # Redis 不可用时 session_id 无效
        session_id = None

    result = rag_system.ask_question(
        request.question,
        stream=request.stream,
        session_id=session_id,
        chat_history=chat_history,
    )

    if not request.stream:
        # 存储问答对
        if session_id and session_manager:
            try:
                await session_manager.append(session_id, "user", request.question)
                await session_manager.append(session_id, "assistant", str(result))
            except Exception as e:
                logger.warning(f"存储对话历史失败: {e}")
        return {"answer": result, "stream": False, "session_id": session_id}

    # ---- SSE 流式响应 ----
    async def event_stream():
        full_answer = []
        try:
            for chunk in result:
                full_answer.append(chunk)
                payload = json.dumps({"content": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

            # 流结束后存储问答对
            if session_id and session_manager and full_answer:
                try:
                    answer_text = "".join(full_answer)
                    await session_manager.append(session_id, "user", request.question)
                    await session_manager.append(session_id, "assistant", answer_text)
                except Exception as e:
                    logger.warning(f"存储对话历史失败: {e}")

            # 在 [DONE] 中返回 session_id
            done_payload = json.dumps({"done": True, "session_id": session_id}, ensure_ascii=False)
            yield f"data: {done_payload}\n\n"
        except Exception as e:
            logger.error(f"流式生成出错: {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/categories")
async def get_categories():
    """获取所有分类和难度标签"""
    return {
        "categories": DataPreparationModule.get_supported_categories(),
        "difficulties": DataPreparationModule.get_supported_difficulties(),
        "mapping": DataPreparationModule.CATEGORY_MAPPING,
    }


@app.get("/api/recipes")
async def list_recipes(
    category: str = Query("", description="按分类筛选"),
    difficulty: str = Query("", description="按难度筛选"),
    query: str = Query("", description="额外搜索关键词"),
):
    """按分类/难度浏览食谱"""
    err = _check_ready()
    if err:
        return err

    if category:
        dishes = rag_system.search_by_category(category, query=query or category)
    else:
        dishes = rag_system.search_by_category(query or "家常", query=query or "")

    return {"dishes": dishes, "total": len(dishes)}


@app.get("/api/stats")
async def get_stats():
    """获取知识库统计"""
    err = _check_ready()
    if err:
        return err

    return rag_system.data_module.get_statistics()


@app.get("/api/daily-recommendation")
async def daily_recommendation():
    """每日推荐 — 返回一道今日精选菜品及推荐理由"""
    err = _check_ready()
    if err:
        return err

    try:
        result = rag_system.get_daily_recommendation()
        return result
    except Exception as e:
        logger.error(f"每日推荐生成失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"每日推荐生成失败: {str(e)}"},
        )


# ---------------------------------------------------------------------------
# 静态文件服务（生产环境：后端直接提供前端页面）
# ---------------------------------------------------------------------------
_frontend_dir = (Path(__file__).parent.parent / "frontend").resolve()
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

    @app.get("/")
    async def serve_frontend():
        """提供前端页面"""
        return FileResponse(str(_frontend_dir / "index.html"))


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8899,
        reload=False,
        log_level="info",
    )
