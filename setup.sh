#!/bin/bash

################################################################################
# COMPLETE GPU Setup Script for Photo Validation API
# NVIDIA L40S | Driver 550.163.01 | CUDA 12.4 | Python 3.10.13
#
# Full GPU Acceleration:
#   InsightFace + DeepFace + NudeNet + CLIP + EasyOCR + PyTorch
#
# CRITICAL ORDER:
#   1. TensorFlow first (brings its own CUDA libs)
#   2. onnxruntime-gpu (initially without CUDA - that's OK)
#   3. Face libs (InsightFace, DeepFace, RetinaFace, NudeNet)
#   4. Other deps (opencv, numpy, etc.)
#   5. NVIDIA CUDA libs (OVERWRITE TF's versions)
#   6. UNINSTALL + REINSTALL onnxruntime-gpu (now picks up correct CUDA)
#   7. tf-keras
#   8. PyTorch + TorchVision (cu121)
#   9. CLIP (--no-build-isolation)
#   10. pillow-heif, easyocr, psycopg2, dotenv, boto3
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         COMPLETE GPU Setup - Photo Validation API                   ║
║         NVIDIA L40S | CUDA 12.4 | Python 3.10.13                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${GREEN}System Configuration:${NC}"
echo "  GPU: NVIDIA L40S (46GB VRAM)"
echo "  Driver: 550.163.01"
echo "  CUDA: 12.4"
echo "  Python: 3.10.13"
echo ""
echo -e "${YELLOW}Press ENTER to continue or Ctrl+C to cancel${NC}"
read

# ==================== STEP 1: VERIFY SYSTEM ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 1: System Verification${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found - GPU drivers not installed${NC}"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
echo -e "${GREEN}✓${NC} GPU: $GPU_INFO"

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo -e "${GREEN}✓${NC} CUDA: $CUDA_VERSION"

if [[ ! "$CUDA_VERSION" =~ ^12\. ]]; then
    echo -e "${YELLOW}⚠${NC} Warning: Expected CUDA 12.x, found $CUDA_VERSION"
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.10\. ]]; then
    echo -e "${YELLOW}⚠${NC} Warning: Expected Python 3.10.x, found $PYTHON_VERSION"
    echo -e "${YELLOW}  The script is tested with Python 3.10.13${NC}"
fi

# ==================== STEP 2: CREATE/ACTIVATE VIRTUAL ENVIRONMENT ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 2: Python Virtual Environment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment already exists at ./$VENV_DIR"
    echo -e "${YELLOW}  Activating existing venv...${NC}"
    source "$VENV_DIR/bin/activate"
else
    echo -e "${YELLOW}Creating virtual environment at ./$VENV_DIR ...${NC}"
    python -m venv "$VENV_DIR"
    echo -e "${GREEN}✓${NC} Virtual environment created"
    source "$VENV_DIR/bin/activate"
fi

echo -e "${GREEN}✓${NC} venv activated: $(which python)"
echo -e "${GREEN}✓${NC} Python in venv: $(python --version)"

# Upgrade pip, setuptools (<81 for CLIP pkg_resources), wheel
echo -e "${YELLOW}Upgrading pip, setuptools (<81), wheel...${NC}"
pip install --upgrade pip "setuptools<81" wheel
echo -e "${GREEN}✓${NC} pip upgraded: $(pip --version)"
echo -e "${GREEN}✓${NC} setuptools (<81, for pkg_resources) + wheel upgraded"

# ==================== STEP 3: BACKUP ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 3: Creating Backup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
pip freeze > "$BACKUP_DIR/packages_before.txt" 2>/dev/null || true
echo -e "${GREEN}✓${NC} Backup: $BACKUP_DIR/"

# ==================== STEP 4: CLEAN OLD PACKAGES ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 4: Removing Old/Conflicting Packages${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

pip uninstall -y tensorflow tensorflow-gpu tf-keras 2>/dev/null || true
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
pip uninstall -y insightface deepface retina-face nudenet 2>/dev/null || true
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip uninstall -y clip easyocr 2>/dev/null || true
pip cache purge 2>/dev/null || true
echo -e "${GREEN}✓${NC} Cleanup complete"

# ==================== STEP 5: TENSORFLOW GPU ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 5: Installing TensorFlow 2.20.0 (GPU)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

pip install --no-cache-dir tensorflow==2.20.0
echo -e "${GREEN}✓${NC} TensorFlow 2.20.0 installed"

# Verify TensorFlow GPU
echo "Verifying TensorFlow GPU..."
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'  ✓ TensorFlow GPU: {len(gpus)} device(s) detected')
else:
    print('  ✗ TensorFlow: No GPU detected')
    exit(1)
"
if [ $? -ne 0 ]; then
    echo -e "${RED}TensorFlow GPU verification failed${NC}"
    exit 1
fi

# ==================== STEP 6: ONNX RUNTIME GPU (first install) ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 6: Installing ONNX Runtime GPU 1.20.1 (first install)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}⚠ NOTE: CUDA may not be detected yet - will be fixed in Step 10${NC}"

pip install --no-cache-dir onnxruntime-gpu==1.20.1
echo -e "${GREEN}✓${NC} ONNX Runtime GPU 1.20.1 installed"

# ==================== STEP 7: FACE RECOGNITION LIBRARIES ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 7: Installing Face Recognition Libraries${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}[7.1] InsightFace 0.7.3...${NC}"
pip install --no-cache-dir insightface==0.7.3
echo -e "${GREEN}✓${NC} InsightFace 0.7.3 installed"

echo -e "${YELLOW}[7.2] DeepFace 0.0.93...${NC}"
pip install --no-cache-dir deepface==0.0.93
echo -e "${GREEN}✓${NC} DeepFace 0.0.93 installed"

echo -e "${YELLOW}[7.3] RetinaFace 0.0.17...${NC}"
pip install --no-cache-dir retina-face==0.0.17
echo -e "${GREEN}✓${NC} RetinaFace 0.0.17 installed"

# ==================== STEP 8: NUDENET ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 8: Installing NudeNet 3.4.2 (GPU via ONNX Runtime)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}[8.1] onnx 1.17.0...${NC}"
pip install --no-cache-dir onnx==1.17.0
echo -e "${GREEN}✓${NC} onnx 1.17.0 installed"

echo -e "${YELLOW}[8.2] NudeNet 3.4.2...${NC}"
pip install --no-cache-dir nudenet==3.4.2
echo -e "${GREEN}✓${NC} NudeNet 3.4.2 installed"

# ==================== STEP 9: OTHER DEPENDENCIES ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 9: Installing Other Dependencies${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

pip install --no-cache-dir \
    opencv-python==4.10.0.84 \
    opencv-contrib-python==4.10.0.84 \
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

echo -e "${GREEN}✓${NC} All base dependencies installed"

# ==============================================================================
# STEP 10: FIX ONNX RUNTIME GPU - CUDA SUPPORT
# ==============================================================================
# CRITICAL: TensorFlow installed its own CUDA lib versions which don't work
# with onnxruntime-gpu. We must:
#   1. Install the correct NVIDIA CUDA runtime libraries
#   2. UNINSTALL onnxruntime-gpu
#   3. REINSTALL onnxruntime-gpu so it picks up the correct CUDA libs
# This is the PROVEN working sequence from manual testing.
# ==============================================================================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 10: Fixing ONNX Runtime GPU - CUDA Support${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}⚠ CRITICAL: Installing CUDA libs then reinstalling onnxruntime-gpu${NC}"

echo -e "${YELLOW}[10.1] Installing NVIDIA CUDA runtime libraries...${NC}"
pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12
echo -e "${GREEN}✓${NC} NVIDIA CUDA runtime libraries installed"

echo -e "${YELLOW}[10.2] Uninstalling onnxruntime-gpu...${NC}"
pip uninstall -y onnxruntime-gpu
echo -e "${GREEN}✓${NC} onnxruntime-gpu uninstalled"

echo -e "${YELLOW}[10.3] Reinstalling onnxruntime-gpu 1.20.1 (with CUDA support)...${NC}"
pip install --no-cache-dir onnxruntime-gpu==1.20.1
echo -e "${GREEN}✓${NC} onnxruntime-gpu 1.20.1 reinstalled"

# Verify ONNX Runtime CUDA
echo "Verifying ONNX Runtime CUDA..."
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'  Available providers: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('  ✓ CUDAExecutionProvider available')
else:
    print('  ✗ CUDAExecutionProvider NOT available')
    exit(1)
"
if [ $? -ne 0 ]; then
    echo -e "${RED}ONNX Runtime CUDA verification failed!${NC}"
    echo -e "${RED}CUDAExecutionProvider not detected after reinstall.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ ONNX Runtime GPU now has CUDAExecutionProvider!${NC}"

# ==================== STEP 11: TF-KERAS ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 11: Installing tf-keras (required by DeepFace)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

pip install --no-cache-dir tf-keras
echo -e "${GREEN}✓${NC} tf-keras installed"

# ==================== STEP 12: PYTORCH + TORCHVISION (CUDA 12.1) ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 12: Installing PyTorch + TorchVision (CUDA 12.1)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
echo -e "${GREEN}✓${NC} PyTorch + TorchVision (CUDA 12.1) installed"

# Verify PyTorch CUDA
echo "Verifying PyTorch CUDA..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  ✓ PyTorch CUDA: {torch.cuda.get_device_name(0)}')
else:
    print('  ⚠ PyTorch: CUDA not available (CLIP will use CPU)')
"

# ==================== STEP 13: OPENAI CLIP ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 13: Installing OpenAI CLIP${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# --no-build-isolation: uses venv's setuptools (<81) which has pkg_resources
pip install --no-cache-dir --no-build-isolation git+https://github.com/openai/CLIP.git
echo -e "${GREEN}✓${NC} OpenAI CLIP installed"

# ==================== STEP 14: PILLOW-HEIF ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 14: Installing Pillow-HEIF (HEIC support)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

pip install --no-cache-dir pillow-heif
echo -e "${GREEN}✓${NC} Pillow-HEIF installed"

# ==================== STEP 15: EASYOCR ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 15: Installing EasyOCR (PII Detection)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

pip install --no-cache-dir easyocr
echo -e "${GREEN}✓${NC} EasyOCR installed"

# ==================== STEP 16: DATABASE & AWS ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 16: Installing Database & AWS Dependencies${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}[16.1] psycopg2-binary (PostgreSQL)...${NC}"
pip install --no-cache-dir psycopg2-binary
echo -e "${GREEN}✓${NC} psycopg2-binary installed"

echo -e "${YELLOW}[16.2] python-dotenv + boto3...${NC}"
pip install --no-cache-dir python-dotenv boto3
echo -e "${GREEN}✓${NC} python-dotenv + boto3 installed"

pip freeze > "$BACKUP_DIR/packages_after.txt"

# ==================== STEP 17: REAL-ESRGAN (Image Enhancement) ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 17: Installing Real-ESRGAN + BasicSR (GPU Image Enhancement)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}[17.1] basicsr (RRDBNet architecture)...${NC}"
pip install --no-cache-dir basicsr
echo -e "${GREEN}✓${NC} basicsr installed"

echo -e "${YELLOW}[17.2] realesrgan...${NC}"
pip install --no-cache-dir realesrgan
echo -e "${GREEN}✓${NC} realesrgan installed"

# Download RealESRGAN x4plus model weights
echo -e "${YELLOW}[17.3] Downloading RealESRGAN_x4plus model weights...${NC}"
WEIGHTS_DIR="weights"
mkdir -p "$WEIGHTS_DIR"
if [ ! -f "$WEIGHTS_DIR/RealESRGAN_x4plus.pth" ]; then
    wget -q -O "$WEIGHTS_DIR/RealESRGAN_x4plus.pth" \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    echo -e "${GREEN}✓${NC} RealESRGAN_x4plus.pth downloaded to $WEIGHTS_DIR/"
else
    echo -e "${GREEN}✓${NC} RealESRGAN_x4plus.pth already exists"
fi

# Verify RealESRGAN
echo "Verifying RealESRGAN..."
python3 -c "
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch
print(f'   ✓ RRDBNet imported successfully')
print(f'   ✓ RealESRGANer imported successfully')
if torch.cuda.is_available():
    print(f'   ✓ Will use GPU: {torch.cuda.get_device_name(0)}')
else:
    print(f'   ⚠ Will use CPU (slower)')
"
echo -e "${GREEN}✓${NC} Real-ESRGAN ready"

# ==================== STEP 18: GPU ENVIRONMENT CONFIG ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 18: Configuring GPU Environment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

if [ -d "/usr/local/cuda-12.4" ]; then
    CUDA_PATH="/usr/local/cuda-12.4"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
else
    CUDA_PATH="/usr/local/cuda-12.4"
fi

cat > gpu_env_config.sh << EOF
#!/bin/bash
# GPU Environment for CUDA 12.4 - Photo Validation API

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2
export TF_CPP_MIN_LOG_LEVEL=2

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH
export PATH=${CUDA_PATH}/bin:\$PATH

export ORT_TENSORRT_FP16_ENABLE=1

echo "✓ GPU Environment configured (CUDA 12.4)"
EOF

chmod +x gpu_env_config.sh
echo -e "${GREEN}✓${NC} gpu_env_config.sh created"

cat > start_gpu_api.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "✗ ERROR: venv not found at $SCRIPT_DIR/venv"
    echo "  Run setup.sh first to create the virtual environment"
    exit 1
fi

source gpu_env_config.sh
echo "Starting Photo Validation API with full GPU acceleration..."
python hybrid_dev_api.py
EOF

chmod +x start_gpu_api.sh
echo -e "${GREEN}✓${NC} start_gpu_api.sh created"

# ==================== STEP 19: FINAL VERIFICATION ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 19: Final GPU Verification (All Libraries)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

source gpu_env_config.sh

python3 << 'PYEOF'
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("FINAL GPU VERIFICATION - ALL LIBRARIES")
print("="*60)

all_pass = True

# 1. TensorFlow
print("\n1. TensorFlow GPU:")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        print(f"   ✓ Version: {tf.__version__}")
        print(f"   ✓ Device: {details.get('device_name', 'Unknown')}")
        print(f"   ✓ Compute Capability: {details.get('compute_capability', 'Unknown')}")
    else:
        print("   ✗ No GPUs detected")
        all_pass = False
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

# 2. ONNX Runtime
print("\n2. ONNX Runtime GPU:")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"   Providers: {providers}")
    if 'CUDAExecutionProvider' in providers:
        print(f"   ✓ CUDAExecutionProvider available")
    else:
        print(f"   ✗ CUDAExecutionProvider NOT available")
        all_pass = False
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

# 3. InsightFace
print("\n3. InsightFace:")
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"   ✓ Initialized with GPU")
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

# 4. DeepFace
print("\n4. DeepFace:")
try:
    from deepface import DeepFace
    import deepface
    print(f"   ✓ Version: {deepface.__version__}")
    if gpus:
        print(f"   ✓ Will use TensorFlow GPU")
    else:
        print(f"   ⚠ Will use CPU")
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

# 5. RetinaFace
print("\n5. RetinaFace:")
try:
    from retinaface import RetinaFace
    print(f"   ✓ Imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

# 6. NudeNet
print("\n6. NudeNet (NSFW Detection):")
try:
    from nudenet import NudeDetector
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print(f"   ✓ Will use GPU (ONNX Runtime CUDA)")
    else:
        print(f"   ⚠ Will use CPU")
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

# 7. PyTorch
print("\n7. PyTorch:")
try:
    import torch
    print(f"   ✓ Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA Version: {torch.version.cuda}")
    else:
        print(f"   ⚠ CUDA not available (will use CPU)")
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

# 8. CLIP
print("\n8. OpenAI CLIP:")
try:
    import clip
    models = clip.available_models()
    print(f"   ✓ Available models: {models}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

# 9. EasyOCR
print("\n9. EasyOCR:")
try:
    import easyocr
    print(f"   ✓ Imported successfully")
    if torch.cuda.is_available():
        print(f"   ✓ Will use GPU")
    else:
        print(f"   ⚠ Will use CPU")
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

# 10. Pillow-HEIF
print("\n10. Pillow-HEIF:")
try:
    import pillow_heif
    print(f"   ✓ HEIC/HEIF support available")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 11. Database & AWS
print("\n11. Database & AWS:")
try:
    import psycopg2
    print(f"   ✓ psycopg2 (PostgreSQL)")
except:
    print(f"   ✗ psycopg2 missing")
    all_pass = False

try:
    import boto3
    print(f"   ✓ boto3 (AWS S3/Rekognition)")
except:
    print(f"   ✗ boto3 missing")
    all_pass = False

try:
    from dotenv import load_dotenv
    print(f"   ✓ python-dotenv (.env config)")
except:
    print(f"   ✗ python-dotenv missing")
    all_pass = False

# 12. FastAPI
print("\n12. FastAPI:")
try:
    import fastapi
    import uvicorn
    print(f"   ✓ FastAPI {fastapi.__version__}")
    print(f"   ✓ Uvicorn installed")
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

# 13. Real-ESRGAN
print("\n13. Real-ESRGAN (Image Enhancement):")
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    print(f"   ✓ basicsr + RRDBNet imported")
    print(f"   ✓ RealESRGANer imported")
    if torch.cuda.is_available():
        print(f"   ✓ Will use GPU for 4x upscaling")
    else:
        print(f"   ⚠ Will use CPU (slower)")
except Exception as e:
    print(f"   ✗ Error: {e}")
    all_pass = False

print("\n" + "="*60)
if all_pass:
    print("✓ ALL CHECKS PASSED - READY FOR PRODUCTION")
else:
    print("⚠ SOME CHECKS FAILED - Review errors above")
print("="*60 + "\n")
PYEOF

# ==================== SUMMARY ====================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}INSTALLATION COMPLETE!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${BLUE}Install Order (proven working sequence):${NC}"
echo "  Step  2: Python venv created/activated"
echo "  Step  5: TensorFlow 2.20.0 (GPU)"
echo "  Step  6: ONNX Runtime GPU 1.20.1 (first install)"
echo "  Step  7: InsightFace, DeepFace, RetinaFace"
echo "  Step  8: NudeNet 3.4.2"
echo "  Step  9: opencv, numpy, scipy, etc."
echo "  Step 10: NVIDIA CUDA libs → uninstall → reinstall onnxruntime-gpu  ← KEY FIX"
echo "  Step 11: tf-keras"
echo "  Step 12: PyTorch + TorchVision (CUDA 12.1)"
echo "  Step 13: OpenAI CLIP (--no-build-isolation)"
echo "  Step 14: Pillow-HEIF"
echo "  Step 15: EasyOCR"
echo "  Step 16: psycopg2, python-dotenv, boto3"
echo "  Step 17: Real-ESRGAN + basicsr (GPU image enhancement)"

echo ""
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo ""
echo "1. Activate virtual environment + GPU environment:"
echo -e "   ${GREEN}source venv/bin/activate && source gpu_env_config.sh${NC}"
echo ""
echo "2. Configure .env file (AWS credentials, DB config, UAT flags):"
echo -e "   ${GREEN}nano .env${NC}"
echo ""
echo "3. Start API (venv is auto-activated by the script):"
echo -e "   ${GREEN}./start_gpu_api.sh${NC}"
echo "   or manually:"
echo -e "   ${GREEN}source venv/bin/activate && source gpu_env_config.sh && python hybrid_dev_api.py${NC}"
echo ""
echo "4. Monitor GPU:"
echo -e "   ${GREEN}watch -n 1 nvidia-smi${NC}"
echo ""
echo "5. Test API:"
echo -e "   ${GREEN}curl http://localhost:8000/health${NC}"
echo ""
echo -e "${GREEN}Ready to go!${NC}"
echo ""
