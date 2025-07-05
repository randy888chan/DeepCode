FROM ubuntu:latest

ENV CONDA_ENV_NAME=deepcode
ENV PYTHON_VERSION=3.13
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    openssh-server \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# 自动检测架构并下载对应的Miniconda版本
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_ARCH="x86_64"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINICONDA_ARCH="aarch64"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# 创建环境
RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y

# 使用conda run来在指定环境中运行命令
RUN conda run -n ${CONDA_ENV_NAME} conda install -y numpy pandas matplotlib jupyter ipython
RUN conda run -n ${CONDA_ENV_NAME} pip install requests

# 安装你的包列表
RUN conda run -n ${CONDA_ENV_NAME} pip install \
    mcp-agent \
    mcp-server-git \
    anthropic \
    streamlit \
    nest_asyncio \
    pathlib2 \
    asyncio-mqtt \
    'aiohttp>=3.8.0' \
    'aiofiles>=0.8.0' \
    'PyPDF2>=2.0.0' \
    docling

# 设置默认激活环境（这里可以用conda activate）
RUN conda init bash && \
    echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc

WORKDIR /paper2code
EXPOSE 8501
CMD ["/bin/bash"]