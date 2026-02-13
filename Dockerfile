###############################################################################
# OPTIMIZED MULTI-STAGE DOCKERFILE
# Photo Validation API - GPU Accelerated
# Target: NVIDIA L40S | CUDA 12.4 | Python 3.10
#
# INSTALL ORDER MATCHES setup.sh EXACTLY (proven working sequence):
#   Step 5:  TensorFlow 2.20.0 (brings its own CUDA libs)
#   Step 6:  onnxruntime-gpu 1.20.1 (first install - CUDA not yet working)
#   Step 7:  InsightFace, DeepFace, RetinaFace
#   Step 8:  onnx 1.17.0 + NudeNet 3.4.2
#   Step 9:  opencv, numpy, scipy, Pillow, scikit-image, fastapi, uvicorn, etc.
#   Step 10: NVIDIA CUDA libs → UNINSTALL onnxruntime-gpu → REINSTALL (CUDA fix)
#   Step 11: tf-keras
#   Step 12: PyTorch + TorchVision (cu121) ← MUST be AFTER CUDA fix
#   Step 13: OpenAI CLIP (--no-build-isolation)
#   Step 14: pillow-heif
#   Step 15: EasyOCR
#   Step 16: psycopg2, python-dotenv, boto3
#   Step 17: RealESRGAN + basicsr patch
#
# OPTIMIZATION STRATEGIES:
#   1. Multi-stage build (builder + runtime) to drop build tools (~800MB saved)
#   2. --no-cache-dir everywhere (PIP_NO_CACHE_DIR=1)
#   3. opencv-python-headless instead of full opencv (~200MB saved)
#   4. Runtime stage has no gcc/cmake/dev headers
#   5. Model weights downloaded as cacheable layer
#   6. App code COPY is last layer (fastest rebuilds)
#
# EXPECTED SIZE: ~6-7GB (CUDA runtime alone is ~3.5GB - unavoidable for GPU)
###############################################################################

# ==============================================================================
# STAGE 1: BUILDER - Install all packages in proven order from setup.sh
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python 3.10 + build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libheif-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip + setuptools (<81 for CLIP pkg_resources compatibility)
RUN pip install --upgrade pip "setuptools<81" wheel

# ==================== Step 5: TensorFlow GPU ====================
RUN pip install tensorflow==2.20.0

# ==================== Step 6: ONNX Runtime GPU (first install) ====================
# NOTE: CUDA will NOT be detected yet - fixed in Step 10
RUN pip install onnxruntime-gpu==1.20.1

# ==================== Step 7: Face Recognition Libraries ====================
RUN pip install insightface==0.7.3
RUN pip install deepface==0.0.93
RUN pip install retina-face==0.0.17

# ==================== Step 8: NudeNet (onnx first, then nudenet) ====================
RUN pip install onnx==1.17.0
RUN pip install nudenet==3.4.2

# ==================== Step 9: Other Dependencies ====================
# Using opencv-python-headless instead of opencv-python to save ~200MB
RUN pip install \
    opencv-python-headless==4.10.0.84 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    Pillow==11.0.0 \
    scikit-image==0.24.0 \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    python-multipart==0.0.17 \
    pydantic==2.10.3 \
    requests==2.32.3 \
    tqdm==4.67.1

# ==================== Step 10: FIX ONNX Runtime GPU - CUDA SUPPORT ====================
# CRITICAL ORDER: Install CUDA libs → UNINSTALL onnxruntime-gpu → REINSTALL
# TensorFlow installed its own CUDA lib versions which conflict with onnxruntime.
# This proven sequence ensures CUDAExecutionProvider is available.
RUN pip install \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12
RUN pip uninstall -y onnxruntime-gpu
RUN pip install onnxruntime-gpu==1.20.1

# ==================== Step 11: tf-keras (required by DeepFace) ====================
RUN pip install tf-keras

# ==================== Step 12: PyTorch + TorchVision (CUDA 12.1) ====================
# MUST be after Step 10 CUDA fix - this is the proven working order
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# ==================== Step 13: OpenAI CLIP ====================
# --no-build-isolation: uses venv's setuptools (<81) which has pkg_resources
RUN pip install --no-build-isolation git+https://github.com/openai/CLIP.git

# ==================== Step 14: Pillow-HEIF (HEIC support) ====================
RUN pip install pillow-heif

# ==================== Step 15: EasyOCR (PII Detection) ====================
RUN pip install easyocr

# ==================== Step 16: Database & AWS ====================
RUN pip install psycopg2-binary
RUN pip install python-dotenv boto3

# ==================== Step 17: RealESRGAN + BasicSR ====================
RUN pip install realesrgan

# Fix basicsr torchvision compatibility (functional_tensor → functional)
RUN BASICSR_LOC=$(python -c "import basicsr; import os; print(os.path.dirname(basicsr.__file__))") && \
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    "$BASICSR_LOC/data/degradations.py" || true

# ==============================================================================
# STAGE 2: RUNTIME - Slim final image (no build tools)
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # GPU environment (matches gpu_env_config.sh)
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_GPU_THREAD_MODE=gpu_private \
    TF_GPU_THREAD_COUNT=2 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=0 \
    ORT_TENSORRT_FP16_ENABLE=1 \
    # App config
    LOG_DIR=/app/logs \
    WEIGHTS_DIR=/app/weights

# Install only runtime dependencies (no gcc, cmake, dev headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libheif1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy virtualenv from builder (all packages pre-installed in correct order)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory
WORKDIR /app

# Download model weights (cacheable layer)
RUN mkdir -p /app/weights && \
    wget -q -O /app/weights/RealESRGAN_x4plus.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Create log directory
RUN mkdir -p /app/logs

# Copy application code (last layer = fastest rebuilds when code changes)
COPY hybrid_dev_api.py hybrid_dev.py image_processor.py db_handler.py ./

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/docs || exit 1

EXPOSE 8001

CMD ["uvicorn", "hybrid_dev_api:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
