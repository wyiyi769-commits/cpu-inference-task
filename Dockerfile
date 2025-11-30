# 使用官方轻量级 Python 镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装必要的依赖
RUN pip install --no-cache-dir numpy

# 将脚本复制到镜像中
COPY inference_task.py .

# 设置容器启动命令
CMD ["python", "-u", "inference_task.py"]
