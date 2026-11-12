# syntax=docker/dockerfile:1.6
# Docker đóng gói ứng dụng Demo 3 tab (QC + Xử lý ảnh cổ điển + ResNet-18 & Grad-CAM)
# Mặc định cài bản PyTorch CPU gọn nhẹ (~850 MB, chạy mọi máy không cần GPU rời).
# Nếu cần bản CUDA 12.6: docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 -t xla-ham10000 .

FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Tách lớp cài đặt thư viện để tận dụng Docker layer cache tối đa
RUN pip install --index-url "${TORCH_INDEX_URL}" torch>=2.4 torchvision>=0.19 && \
    pip install \
        "numpy>=1.26,<3.0" \
        "pandas>=2.2" \
        "scikit-learn>=1.5" \
        "scipy>=1.13" \
        "matplotlib>=3.9" \
        "seaborn>=0.13" \
        "pillow>=10.0" \
        "opencv-python-headless>=4.10" \
        "streamlit>=1.40"

COPY --chown=appuser:appuser .streamlit/ /app/.streamlit/
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser samples/ /app/samples/
COPY --chown=appuser:appuser demo_assets/ /app/demo_assets/
COPY --chown=appuser:appuser app.py /app/app.py

# Tự động nhúng baseline_seed42.pt vào image nếu có sẵn trên máy build; vẫn build thành công trên bản clone sạch
RUN --mount=type=bind,source=.,target=/ctx \
    mkdir -p /app/outputs/checkpoints && \
    if [ -f /ctx/outputs/checkpoints/baseline_seed42.pt ]; then \
        cp /ctx/outputs/checkpoints/baseline_seed42.pt /app/outputs/checkpoints/baseline_seed42.pt; \
    fi && \
    chown -R appuser:appuser /app/outputs

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=3 \
    CMD sh -c 'p="${STREAMLIT_SERVER_BASE_URL_PATH:+/${STREAMLIT_SERVER_BASE_URL_PATH#/}}"; curl --fail --silent "http://127.0.0.1:8501${p}/_stcore/health" >/dev/null || exit 1'

# Lưu ý: Trong nội bộ container cần lắng nghe trên giao diện mạng container để port-mapping từ 127.0.0.1 của host chuyển tiếp vào;
# ở phía host (docker-compose.yml / docker run), cổng được khóa chặt vào 127.0.0.1:8501:8501.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
