"""
尝尝咸淡 RAG 系统 — FastAPI 后端服务
"""
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Generator, Optional

# 确保可以导入同目录下的模块
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
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
_startup_ok: bool = False
_startup_error: str = ""


@app.on_event("startup")
def startup():
    global rag_system, _startup_ok, _startup_error
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
        _startup_ok = True
        print("[OK] RAG 系统启动完成!\n", flush=True)
    except Exception as e:
        _startup_error = str(e)
        _startup_ok = False
        rag_system = None
        traceback.print_exc()
        print(f"\n[ERROR] RAG 系统启动失败: {e}\n", flush=True)
        logger.error(f"RAG 系统启动失败: {e}")


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    stream: bool = True


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


@app.post("/api/ask")
async def ask_recipe(request: AskRequest):
    """智能问答接口 — 支持 SSE 流式与非流式两种模式"""
    err = _check_ready()
    if err:
        return err

    result = rag_system.ask_question(request.question, stream=request.stream)

    if not request.stream:
        return {"answer": result, "stream": False}

    # SSE 流式响应
    async def event_stream():
        try:
            for chunk in result:
                payload = json.dumps({"content": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
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
