"""
索引构建模块
"""

import logging
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称
            index_save_path: 索引保存路径
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()
    
    def setup_embeddings(self):
        """初始化嵌入模型（优先使用本地缓存，避免联网）"""
        import os

        # 阻止 HuggingFace Hub 发起网络请求，强制使用本地缓存
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

        # 解析模型的实际本地路径
        local_model_path = self._resolve_local_model_path(self.model_name)

        if local_model_path:
            logger.info(f"正在从本地加载嵌入模型: {local_model_path}")
        else:
            logger.info(f"正在初始化嵌入模型: {self.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=local_model_path or self.model_name,
            model_kwargs={
                'device': 'cpu',
                # 如果使用本地路径，token 等配置文件也会从本地读取
            },
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info("嵌入模型初始化完成")

    @staticmethod
    def _resolve_local_model_path(model_name: str) -> str | None:
        """
        将 HF 模型名解析为本地缓存路径，避免每次启动联网验证。

        查找顺序：
        1. 直接路径（已经是本地路径）
        2. 项目本地: code/models/{model_name}/
        3. HF 缓存目录: ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}
        """
        from pathlib import Path

        # 1. 已经是本地路径
        if Path(model_name).exists():
            return str(Path(model_name).resolve())

        # 2. 查找项目本地（优先，确保项目可移植）
        local_path = Path(__file__).parent.parent / 'models' / model_name
        if local_path.exists():
            return str(local_path.resolve())

        # 3. 查找 HF 缓存
        org, _, name = model_name.partition('/')
        if org and name:
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
            model_cache = cache_dir / f'models--{org}--{name}'
            snapshots = model_cache / 'snapshots'
            if snapshots.exists():
                # 取最新的 snapshot
                snapshot_dirs = sorted(snapshots.iterdir(),
                                       key=lambda x: x.stat().st_mtime,
                                       reverse=True)
                for sd in snapshot_dirs:
                    if sd.is_dir() and (sd / 'model.safetensors').exists():
                        return str(sd)

        return None
    
    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            
        Returns:
            FAISS向量存储对象
        """
        logger.info("正在构建FAISS向量索引...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        # 构建FAISS向量存储
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return self.vectorstore
    
    def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引添加新文档
        
        Args:
            new_chunks: 新的文档块列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        self.vectorstore.add_documents(new_chunks)
        logger.info("新文档添加完成")

    def save_index(self):
        """
        保存向量索引到配置的路径
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到: {self.index_save_path}")
    
    def load_index(self):
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")
        
        return self.vectorstore.similarity_search(query, k=k)
