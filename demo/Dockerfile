# =================================================================
# STAGE 1: BUILDER - 編譯 TA-Lib C 函式庫與安裝 Python Wheels
# =================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# 安裝編譯所需之系統工具鏈與 C 函式庫標頭檔
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    wget \
    curl \
    default-libmysqlclient-dev \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 下載官方 TA-Lib 0.4.0 C 原始碼並進行編譯與系統安裝
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 建立獨立虛擬環境 /opt/venv 並安裝專案分層依賴
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# 設定 CFLAGS 抑制 GCC 14+ 對 TA-Lib Cython C 代碼的 const 指標不相容警告
ENV CFLAGS="-Wno-error=incompatible-pointer-types -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall"

# 複製分層 requirements
COPY requirements/ /build/requirements/
COPY requirements.txt /build/

# 分層安裝依賴以最大化 Docker Layer Cache
# 步驟 1：先安裝 numpy 確保 TA-Lib 使用 NumPy 1.x C 標頭檔編譯
RUN pip install --no-cache-dir numpy==1.26.4

# 步驟 2：單獨安裝 TA-Lib，停用 build isolation 以使用已安裝的 NumPy 1.x
RUN pip install --no-cache-dir --no-build-isolation TA-Lib==0.4.28

# 步驟 2.5：獨立安裝 PyTorch 2.5.1 CPU 版本 (避免與 TensorFlow CUDA 12.3 之 nvidia 套件衝突，並滿足 Transformers 需求)
RUN pip install --no-cache-dir torch==2.5.1 sympy==1.13.1 --index-url https://download.pytorch.org/whl/cpu

# 步驟 3：安裝其餘分層依賴
RUN pip install --no-cache-dir -r /build/requirements/base.txt && \
    pip install --no-cache-dir -r /build/requirements/ml.txt && \
    pip install --no-cache-dir -r /build/requirements/scraping.txt && \
    pip install --no-cache-dir -r /build/requirements/production.txt

# =================================================================
# STAGE 2: RUNTIME - 輕量化安全執行環境 (Non-Root)
# =================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# 安裝最小運行時依賴：MySQL Client 函式庫、OpenCV 執行庫 (libgl1, libglib) 與網路除錯工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-mysql-client \
    libgl1 \
    libglib2.0-0 \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# 自 Builder 複製 TA-Lib 編譯產出之動態連結函式庫與標頭檔
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/
RUN ldconfig

# 自 Builder 複製完整虛擬環境
COPY --from=builder /opt/venv /opt/venv

# 建立非特權使用者 djangouser (UID 1000) 防禦容器破殼提權攻擊
RUN groupadd -r djangouser && useradd -r -u 1000 -g djangouser -m -d /home/djangouser djangouser

# 建立應用程式主目錄與持久化目錄並分配權限
WORKDIR /app
RUN mkdir -p /app/staticfiles /app/media && \
    chown -R djangouser:djangouser /app

# 複製專案代碼與入口腳本
COPY --chown=djangouser:djangouser . /app/
RUN chmod +x /app/entrypoint.sh

# 切換為非特權使用者運行進程
USER djangouser

EXPOSE 8000

# 入口腳本與預設 ASGI 啟動指令 (Gunicorn + UvicornWorker)
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "demo.asgi:application", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "30", "--graceful-timeout", "30", "--max-requests", "1000", "--max-requests-jitter", "50", "--access-logfile", "-", "--error-logfile", "-"]
