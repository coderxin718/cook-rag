"""
RAG系统主程序
"""

import os
import sys
import json
import re
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)
from lunar_utils import format_date_context

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or RAGConfig()
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 基准目录（code/），用于相对路径解析
        _code_dir = Path(__file__).parent

        # 每日推荐缓存（同一天返回相同结果）
        self._daily_cache: dict = {}
        self._history_path = _code_dir / "recommendation_history.json"

        # 将相对路径解析为基于 code/ 目录的绝对路径，确保无论从何处运行都能找到文件
        if not Path(self.config.data_path).is_absolute():
            self.config.data_path = str((_code_dir / self.config.data_path).resolve())
        if not Path(self.config.index_save_path).is_absolute():
            self.config.index_save_path = str((_code_dir / self.config.index_save_path).resolve())

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")
    
    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n[问题] 用户问题: {question}")

        # 0. 意图守卫 - 拦截与烹饪无关的问题
        print("[守卫] 意图识别...")
        try:
            allowed = self.generation_module.intent_guard(question)
        except Exception as e:
            logger.warning(f"意图守卫调用失败，默认放行: {e}")
            allowed = True

        if not allowed:
            print("[拒绝] 与烹饪无关的问题")
            msg = "小厨只懂做菜，不懂这个哦～"
            if stream:
                return (chunk for chunk in [msg])
            return msg

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"[路由] 查询类型: {route_type}")

        # 2. 智能查询重写（推荐类保持原样，做法类允许重写）
        if route_type == 'recommend':
            rewritten_query = question
            print(f"[查询] 推荐类保持原样: {question}")
        else:
            print("[查询] 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)

        # 3. 检索相关子块（自动应用元数据过滤）
        print("[检索] 搜索相关文档...")
        filters = self._extract_filters_from_query(question)
        relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)
        if filters:
            print(f"应用过滤条件: {filters}")
            filtered = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=self.config.top_k)
            if filtered:
                relevant_chunks = filtered
            else:
                print(f"过滤后无结果，回退到无过滤检索")

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get('dish_name', '未知菜品')
                section_title = (chunk.metadata.get('三级标题')
                              or chunk.metadata.get('二级标题')
                              or chunk.metadata.get('主标题')
                              or None)
                if section_title:
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    preview = chunk.page_content[:30].strip().replace('\n', ' ')
                    chunk_info.append(f"{dish_name}({preview}...)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 0 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            msg = "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
            if stream:
                return (chunk for chunk in [msg])
            return msg

        # 5. 获取完整文档
        print("获取完整文档...")
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

        doc_names = []
        for doc in relevant_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            doc_names.append(dish_name)

        if doc_names:
            print(f"找到文档: {', '.join(doc_names)}")

        # 6. 根据路由类型选择回答方式
        if route_type == 'recommend':
            print("[生成] 推荐列表...")
            answer = self.generation_module.generate_list_answer(question, relevant_docs)
            if stream:
                return (chunk for chunk in [answer])
            return answer
        else:
            print("[生成] 分步骤指导...")
            if stream:
                return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
            else:
                return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
    
    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters
    
    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品
        
        Args:
            category: 菜品分类
            query: 可选的额外查询条件
            
        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")
        
        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}
        
        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)
        
        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)
        
        return dish_names
    
    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 搜索相关文档
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材？", docs)

        return answer
    
    # ========================================================================
    # 每日推荐
    # ========================================================================

    def _load_recommendation_history(self) -> list:
        """加载推荐历史（最多保留30天用于去重）"""
        if not self._history_path.exists():
            return []
        try:
            data = json.loads(self._history_path.read_text(encoding="utf-8"))
            history = data.get("history", [])
            # 只保留最近30天
            cutoff = (date.today() - timedelta(days=30)).isoformat()
            return [h for h in history if h["date"] >= cutoff]
        except Exception:
            return []

    def _save_recommendation_history(self, dish_name: str, date_key: str,
                                     existing: list):
        """追加推荐记录并写回磁盘"""
        existing.append({"date": date_key, "dish_name": dish_name})
        # 只保留最近30天
        cutoff = (date.today() - timedelta(days=30)).isoformat()
        existing = [h for h in existing if h["date"] >= cutoff]
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history_path.write_text(
            json.dumps({"history": existing}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _score_dishes_for_recommendation(dishes: list, season: str,
                                         d: date) -> list:
        """规则引擎：根据季节和工作日给菜品打分排序"""
        import random

        season_boost = {
            "夏季": {"素菜": 3, "甜品": 3, "饮品": 4, "水产": 2, "凉菜": 4,
                    "荤菜": 0, "汤品": 1, "早餐": 2, "主食": 1},
            "冬季": {"荤菜": 4, "汤品": 4, "主食": 3, "素菜": 2, "甜品": 1,
                    "饮品": 0, "半成品": 1},
            "春季": {"素菜": 3, "水产": 3, "早餐": 2, "半成品": 1, "荤菜": 2,
                    "甜品": 2},
            "秋季": {"荤菜": 3, "汤品": 3, "主食": 2, "素菜": 2, "甜品": 2,
                    "半成品": 1},
        }

        is_workday = d.weekday() < 5
        boost = season_boost.get(season, {})

        scored = []
        for dish in dishes:
            score = boost.get(dish["category"], 1)
            diff = dish.get("difficulty", "")
            if is_workday:
                if diff in ("非常简单", "简单"):
                    score += 2
                elif diff == "中等":
                    score += 1
            else:
                if diff in ("困难", "非常困难"):
                    score += 2
            # 微小随机扰动，保证同分时顺序有变化
            score += random.uniform(0, 0.5)
            scored.append({**dish, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _llm_pick_dish(self, dish_names: List[str], date_ctx: dict) -> dict:
        """让 LLM 从候选菜名中选出最适合今天推荐的菜并写推荐理由"""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_template("""\
你是"尝尝咸淡"AI智能食谱助手的主厨。

今天是{display_date}，{lunar_display}，{solar_term}，正值{season}。

请从以下候选菜品中选出最适合今天推荐的一道菜，并用2-3句话写出充满人情味和时令感的推荐理由。
推荐理由要自然地结合季节、节气、工作日/周末等因素，让用户感到贴心。

候选菜品：
{dish_list}

请严格返回以下JSON格式，不要包含markdown代码块或其他额外内容：
{{"dish_name": "菜名", "reason": "推荐理由"}}""")

        dish_list = "\n".join(f"- {name}" for name in dish_names)

        chain = prompt | self.generation_module.llm | StrOutputParser()

        response = chain.invoke({
            "display_date": date_ctx["display_date"],
            "lunar_display": date_ctx["display_lunar"],
            "solar_term": date_ctx["solar_term_context"],
            "season": date_ctx["season"],
            "dish_list": dish_list,
        })

        # 解析 JSON（处理可能的 markdown 包装）
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:])
            if response.endswith("```"):
                response = response[: response.rfind("```")].strip()

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', response)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    result = {}
            else:
                result = {}

        # 校验结果
        dish_name = (result.get("dish_name") or "").strip()
        reason = (result.get("reason") or "").strip()

        if not dish_name or dish_name not in dish_names:
            # LLM 返回的菜名不在候选列表中，用候选第一个做 fallback
            dish_name = dish_names[0]
            reason = reason or f"今天为您推荐一道美味的{dish_name}，正是当季好味道！"

        return {"dish_name": dish_name, "reason": reason}

    def get_daily_recommendation(self) -> dict:
        """
        获取今日推荐菜品。

        Returns:
            {
                "date_context": { ... },   # 日期/农历/节气上下文
                "dish_name": "红烧肉",
                "category": "荤菜",
                "difficulty": "中等",
                "reason": "...",
            }
        """
        today = date.today()
        date_key = today.isoformat()

        # ---- 缓存命中 ----
        if self._daily_cache.get("date") == date_key:
            logger.info(f"每日推荐缓存命中: {date_key}")
            return self._daily_cache["data"]

        logger.info("正在生成每日推荐...")

        # ---- 日期上下文 ----
        ctx = format_date_context(today)

        # ---- 加载历史 & 排除最近推荐过的菜品 ----
        history = self._load_recommendation_history()
        recent_dishes: set = set()
        cutoff = today - timedelta(days=7)
        for entry in history:
            if date.fromisoformat(entry["date"]) >= cutoff:
                recent_dishes.add(entry["dish_name"])

        # ---- 收集所有菜品元数据 ----
        if not self.data_module or not self.data_module.documents:
            raise ValueError("知识库尚未构建，请先调用 build_knowledge_base()")

        all_dishes = []
        for doc in self.data_module.documents:
            name = doc.metadata.get("dish_name", "")
            cat = doc.metadata.get("category", "")
            diff = doc.metadata.get("difficulty", "")
            if name and name not in recent_dishes:
                all_dishes.append({
                    "dish_name": name,
                    "category": cat,
                    "difficulty": diff,
                })

        if not all_dishes:
            # 所有菜都推荐过了，重新开始
            for doc in self.data_module.documents:
                name = doc.metadata.get("dish_name", "")
                if name:
                    all_dishes.append({
                        "dish_name": name,
                        "category": doc.metadata.get("category", ""),
                        "difficulty": doc.metadata.get("difficulty", ""),
                    })

        # ---- 规则引擎初筛 ----
        scored = self._score_dishes_for_recommendation(
            all_dishes, ctx["season"], today
        )
        top_n = min(15, len(scored))
        candidates = scored[:top_n]

        # ---- LLM 精选 ----
        if self.generation_module and self.generation_module.llm:
            try:
                pick = self._llm_pick_dish(
                    [c["dish_name"] for c in candidates], ctx
                )
            except Exception as e:
                logger.warning(f"LLM 精选失败，使用规则引擎结果: {e}")
                pick = {
                    "dish_name": candidates[0]["dish_name"],
                    "reason": f"今天为您推荐一道美味的{candidates[0]['dish_name']}，正是当季好味道！",
                }
        else:
            pick = {
                "dish_name": candidates[0]["dish_name"],
                "reason": f"今天为您推荐一道美味的{candidates[0]['dish_name']}，正是当季好味道！",
            }

        # ---- 补全元数据 ----
        final_dish = pick["dish_name"]
        for d in all_dishes:
            if d["dish_name"] == final_dish:
                pick["category"] = d["category"]
                pick["difficulty"] = d["difficulty"]
                break

        # ---- 保存历史 ----
        self._save_recommendation_history(final_dish, date_key, history)

        # ---- 组装结果 ----
        result = {
            "date_context": ctx,
            "dish_name": pick["dish_name"],
            "category": pick.get("category", ""),
            "difficulty": pick.get("difficulty", ""),
            "reason": pick["reason"],
        }

        # ---- 缓存 ----
        self._daily_cache = {"date": date_key, "data": result}

        logger.info(f"每日推荐: {pick['dish_name']} — {pick['reason'][:40]}...")
        return result

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用尝尝咸淡RAG系统！")



def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
