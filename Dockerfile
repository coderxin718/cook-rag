# 尝尝咸淡 RAG 系统 Docker 镜像
FROM python:3.11-slim

WORKDIR /app

# 换国内 Debian 镜像源（阿里云），加速 apt 下载
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖（FAISS 需要 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存层
COPY code/requirements.txt .
# 换国内 PyPI 镜像 + 先装 CPU 版 PyTorch（比 CUDA 版小 600MB），加速构建
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制源码和模型
COPY code/ ./code/
COPY frontend/ ./frontend/

# 数据目录和索引目录通过 volume 挂载，不在镜像中

WORKDIR /app/code
EXPOSE 8899

CMD ["python", "api_server.py"]
