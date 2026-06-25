# 尝尝咸淡 RAG 系统 Docker 镜像
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（FAISS 需要 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存层
COPY code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码和模型
COPY code/ ./code/
COPY frontend/ ./frontend/

# 数据目录和索引目录通过 volume 挂载，不在镜像中

WORKDIR /app/code
EXPOSE 8899

CMD ["python", "api_server.py"]
