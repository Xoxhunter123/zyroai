FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=10000
ENV MODEL_REPO="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
ENV MODEL_FILE="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
ENV MODEL_ALIAS="Qwen3-Coder-30B-A3B-Instruct"
ENV CTX_SIZE=32768
ENV THREADS=12
ENV DATA_DIR="/data"

# Install build tools + full developer runtime (Python, Node.js, Git, Ripgrep, SQLite)
# so Zyro Agent can create files, run scripts, and execute terminal commands like OpenCode
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    python3 \
    python3-pip \
    python3-venv \
    nodejs \
    npm \
    ripgrep \
    jq \
    sqlite3 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp (llama-server) optimized for 12-vCPU x86_64 Render servers
WORKDIR /opt
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && \
    cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_AVX2=ON \
      -DGGML_FMA=ON \
      -DGGML_F16C=ON \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF && \
    cmake --build /opt/llama.cpp/build --target llama-server -j "$(nproc)"

WORKDIR /app
COPY server.py start_render.sh /app/
RUN sed -i 's/\r$//' /app/start_render.sh && chmod +x /app/start_render.sh

EXPOSE 10000

CMD ["/app/start_render.sh"]
