from fastapi import FastAPI, File, UploadFile, Form, Body, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from enum import Enum
import os
import logging
from logging.handlers import RotatingFileHandler

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
import shutil
import tempfile
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time

# ==================== LOGGING SETUP ====================
LOG_DIR = os.environ.get("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Create logger
logger = logging.getLogger("photo_validation")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_fmt)

# File handler - rotates at 50MB, keeps 5 backups
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "api.log"),
    maxBytes=50 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_fmt = logging.Formatter("[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_fmt)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Import validation system
from hybrid_dev import validate_photo_complete_hybrid, GPU_AVAILABLE, TF_GPU_AVAILABLE, ONNX_GPU_AVAILABLE

# Import image processor for ESRGAN enhancement
from image_processor import process_image_for_sizes, cleanup_processed_images

# Import RetinaFace for face count detection
from retinaface import RetinaFace
from PIL import Image
import cv2

# AWS S3 for primary photo storage
import boto3
from botocore.exceptions import ClientError

# S3 Configuration - Primary photo storage
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "awsphotovalbm")
S3_REGION = os.environ.get("S3_REGION", "ap-south-1")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", None)
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", None)

# Rekognition Face Collections (from existing AWS infrastructure)
REKOGNITION_COLLECTION_1 = os.environ.get("REKOGNITION_COLLECTION_1", "bm_cbs_face_collection")
REKOGNITION_COLLECTION_2 = os.environ.get("REKOGNITION_COLLECTION_2", "bm_cbs_face_collection2")
FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "99.0"))  # Similarity threshold (99% for match)

# UAT Mode - When enabled, disables all WRITE operations to AWS (Rekognition indexing/deletion)
# Set to "true" for UAT testing, "false" or unset for production
UAT_MODE = os.environ.get("UAT_MODE", "true").lower() == "true"

# Skip AWS Checks - When enabled, skips ALL AWS operations (S3, Rekognition)
# Use this when you don't have AWS credentials yet
SKIP_AWS_CHECKS = os.environ.get("SKIP_AWS_CHECKS", "true").lower() == "true"

if SKIP_AWS_CHECKS:
    logger.warning("=" * 70)
    logger.warning("[SKIP AWS] ALL AWS checks are DISABLED (no credentials)")
    logger.warning("[SKIP AWS] S3 lookup, Rekognition celebrity/duplicate checks are skipped")
    logger.warning("[SKIP AWS] Validation will work using local checks only")
    logger.warning("=" * 70)
elif UAT_MODE:
    logger.warning("=" * 70)
    logger.warning("[UAT MODE] Write operations to AWS Rekognition are DISABLED")
    logger.warning("[UAT MODE] Only READ operations (search, celebrity check, duplicate check) are active")
    logger.warning("=" * 70)

# Supported image formats for AWS Rekognition
REKOGNITION_SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

def get_image_bytes_for_rekognition(image_path: str) -> bytes:
    """
    Read image bytes for AWS Rekognition, converting unsupported formats
    (WebP, AVIF, HEIF, etc.) to JPEG automatically.

    AWS Rekognition only supports: JPEG, PNG, BMP, GIF.
    """
    file_ext = os.path.splitext(image_path)[1].lower()

    if file_ext in REKOGNITION_SUPPORTED_FORMATS:
        with open(image_path, 'rb') as f:
            return f.read()
    else:
        logger.info(f"[Rekognition] Converting {file_ext} to JPEG for Rekognition compatibility")
        try:
            from io import BytesIO
            img = Image.open(image_path)
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"[Rekognition] Format conversion failed: {e}, falling back to raw bytes")
            with open(image_path, 'rb') as f:
                return f.read()


# Initialize S3 client
def get_s3_client():
    """Get S3 client with credentials from environment or IAM role"""
    try:
        if AWS_ACCESS_KEY and AWS_SECRET_KEY:
            return boto3.client(
                's3',
                region_name=S3_REGION,
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
        else:
            # Use IAM role or default credentials
            return boto3.client('s3', region_name=S3_REGION)
    except Exception as e:
        logger.error(f"[S3] Error initializing S3 client: {e}")
        return None

# Initialize Rekognition client
def get_rekognition_client():
    """Get Rekognition client with credentials from environment or IAM role"""
    try:
        if AWS_ACCESS_KEY and AWS_SECRET_KEY:
            return boto3.client(
                'rekognition',
                region_name=S3_REGION,
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
        else:
            # Use IAM role or default credentials
            return boto3.client('rekognition', region_name=S3_REGION)
    except Exception as e:
        logger.error(f"[Rekognition] Error initializing client: {e}")
        return None

# Import database handler
from db_handler import (
    initialize_database,
    insert_validation_with_matri_id,
    get_validation_by_id,
    get_validations_by_matri_id,
    get_validations_by_batch_id,
    get_validation_statistics,
    close_all_connections
)

app = FastAPI(
    title="Photo Validation API - Hybrid GPU",
    description="GPU-accelerated photo validation (InsightFace + DeepFace + NudeNet)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, status code, and duration."""
    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    # Extract matri_id from form data if available (log-safe)
    logger.info(f"[{request_id}] --> {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        duration = round(time.time() - start, 3)
        logger.info(f"[{request_id}] <-- {response.status_code} ({duration}s)")
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        duration = round(time.time() - start, 3)
        logger.error(f"[{request_id}] <-- 500 UNHANDLED ({duration}s): {e}")
        raise

executor = ThreadPoolExecutor(max_workers=4)

# ==================== MODELS ====================

class SingleImageResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
    response_time_seconds: Optional[float] = None
    library_usage: Optional[dict] = None
    gpu_info: Optional[dict] = None

class MultiImageResponse(BaseModel):
    success: bool
    message: str
    batch_id: str
    total_images: int
    results: List[dict]
    summary: dict
    response_time_seconds: Optional[float] = None
    library_usage_summary: Optional[dict] = None
    gpu_info: Optional[dict] = None

# ==================== HELPER FUNCTIONS ====================

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    temp_dir = os.path.join(tempfile.gettempdir(), "photo_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    file_extension = os.path.splitext(upload_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    temp_path = os.path.join(temp_dir, unique_filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return temp_path

def cleanup_temp_files(*file_paths):
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                cropped_path = file_path.replace(".", "_cropped_final.")
                if os.path.exists(cropped_path):
                    os.remove(cropped_path)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# ==================== S3 UPLOAD HELPERS ====================

# S3 bucket for processed/validated photos
S3_VALIDATION_BUCKET = os.environ.get("S3_BUCKET_NAME", "bm-photo-validation")

def upload_file_to_s3(file_path: str, s3_key: str, content_type: str = None) -> dict:
    """
    Upload a local file to S3.

    Args:
        file_path: Local file path
        s3_key: S3 object key (e.g., 'approved/BM12345/photo_1080x1080.jpg')
        content_type: MIME type (auto-detected if not provided)

    Returns:
        dict with success, s3_key, error
    """
    result = {"success": False, "s3_key": s3_key, "error": None}

    try:
        s3_client = get_s3_client()
        if s3_client is None:
            result["error"] = "S3 client not available"
            return result

        if content_type is None:
            ext = os.path.splitext(file_path)[1].lower()
            content_type_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
                ".gif": "image/gif", ".bmp": "image/bmp",
            }
            content_type = content_type_map.get(ext, "application/octet-stream")

        with open(file_path, "rb") as f:
            s3_client.put_object(
                Bucket=S3_VALIDATION_BUCKET,
                Key=s3_key,
                Body=f.read(),
                ContentType=content_type,
            )

        result["success"] = True
        logger.info(f"[S3] Uploaded: s3://{S3_VALIDATION_BUCKET}/{s3_key}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[S3] Upload failed for {s3_key}: {e}")

    return result


def upload_bytes_to_s3(data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> dict:
    """Upload raw bytes to S3."""
    result = {"success": False, "s3_key": s3_key, "error": None}

    try:
        s3_client = get_s3_client()
        if s3_client is None:
            result["error"] = "S3 client not available"
            return result

        s3_client.put_object(
            Bucket=S3_VALIDATION_BUCKET,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )

        result["success"] = True
        logger.info(f"[S3] Uploaded: s3://{S3_VALIDATION_BUCKET}/{s3_key}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[S3] Upload failed for {s3_key}: {e}")

    return result




def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def format_validation_result(result: dict, image_filename: str) -> dict:
    result = convert_numpy_types(result)
    
    final_decision = result["final_decision"]
    if final_decision == "APPROVE":
        final_status = "ACCEPTED"
    elif final_decision == "REJECT":
        final_status = "REJECTED"
    elif final_decision == "SUSPEND":
        final_status = "SUSPENDED"
    elif final_decision == "MANUAL_REVIEW":
        final_status = "MANUAL_REVIEW"
    else:
        final_status = "ERROR"
    
    library_usage = None
    if result.get("stage2") and result["stage2"].get("library_usage"):
        library_usage = {
            "insightface": result["stage2"]["library_usage"]["insightface"],
            "deepface": result["stage2"]["library_usage"]["deepface"],
            "nudenet": "GPU" if ONNX_GPU_AVAILABLE else "CPU",
            "gpu_used": result["stage2"].get("gpu_used", False)
        }
    
    return {
        "image_filename": image_filename,
        "validation_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "photo_type": result.get("photo_type"),
        "final_status": final_status,
        "final_reason": result["final_reason"],
        "final_action": result["final_action"],
        "final_decision": final_decision,
        "image_was_cropped": result.get("image_was_cropped", False),
        "cropped_image_base64": result.get("cropped_image_base64"),
        "checklist_summary": result.get("checklist_summary"),
        "stage1_checks": result["stage1"]["checks"],
        "stage2_checks": result.get("stage2", {}).get("checks", {}) if result.get("stage2") else None,
        "library_usage": library_usage,
        "validation_approach": "hybrid" if library_usage else "stage1_only"
    }

def save_validation_to_db(validation_data: dict, matri_id: str, batch_id: str = None,
                          response_time: float = None, gpu_info: dict = None,
                          product_name: str = None):
    """Save validation result to PostgreSQL database"""
    try:
        success = insert_validation_with_matri_id(
            validation_data=validation_data,
            matri_id=matri_id,
            batch_id=batch_id,
            response_time=response_time,
            gpu_info=gpu_info,
            product_name=product_name
        )
        if success:
            logger.info(f"[DB] Saved validation {validation_data.get('validation_id')} for matri_id: {matri_id}")
        else:
            logger.error(f"[DB] Failed to save validation {validation_data.get('validation_id')}")
        return success
    except Exception as e:
        logger.error(f"[DB] Error saving validation to database: {e}")
        return False

def search_face_in_rekognition(image_path: str, matri_id: str) -> dict:
    """
    Search for a face in Rekognition collections using an image.

    Returns dict with:
        - found: bool - whether a matching face was found
        - same_id_matches: list - matches with same matri_id
        - diff_id_matches: list - matches with different matri_id (potential duplicates)
        - face_id: str - the face ID if found
        - s3_key: str - the S3 key of the matched image
    """
    result = {
        "found": False,
        "same_id_matches": [],
        "diff_id_matches": [],
        "face_id": None,
        "s3_key": None,
        "error": None
    }

    try:
        rekognition = get_rekognition_client()
        if rekognition is None:
            result["error"] = "Rekognition client not available"
            return result

        # Read image bytes (convert WebP/AVIF/HEIF to JPEG if needed)
        image_bytes = get_image_bytes_for_rekognition(image_path)

        # Search in both collections
        all_matches = []

        for collection_id in [REKOGNITION_COLLECTION_1, REKOGNITION_COLLECTION_2]:
            try:
                response = rekognition.search_faces_by_image(
                    CollectionId=collection_id,
                    Image={'Bytes': image_bytes},
                    MaxFaces=10,
                    FaceMatchThreshold=FACE_MATCH_THRESHOLD
                )

                if 'FaceMatches' in response:
                    all_matches.extend(response['FaceMatches'])

            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'InvalidParameterException':
                    # No face detected in image
                    logger.info(f"[Rekognition] No face detected in image for search")
                elif error_code == 'ResourceNotFoundException':
                    logger.info(f"[Rekognition] Collection {collection_id} not found")
                else:
                    logger.error(f"[Rekognition] Error searching collection {collection_id}: {e}")
                continue

        # Process matches
        for match in all_matches:
            similarity = match['Similarity']
            external_id = match['Face'].get('ExternalImageId', '')
            face_id = match['Face']['FaceId']

            # ExternalImageId format is typically: matri_id or matri_id_suffix
            matched_matri_id = external_id.partition("_")[0] if external_id else ""

            if matched_matri_id == matri_id:
                # Same user - this is their existing indexed face
                if similarity >= 99.0:
                    result["same_id_matches"].append({
                        "face_id": face_id,
                        "external_id": external_id,
                        "similarity": similarity
                    })
                    if not result["found"]:
                        result["found"] = True
                        result["face_id"] = face_id
                        result["s3_key"] = external_id
            else:
                # Different user - potential duplicate
                if similarity >= 99.0:
                    result["diff_id_matches"].append({
                        "matched_matri_id": matched_matri_id,
                        "face_id": face_id,
                        "external_id": external_id,
                        "similarity": similarity
                    })

        if result["found"]:
            logger.info(f"[Rekognition] Found existing face for {matri_id} in collection")
        else:
            logger.info(f"[Rekognition] No existing face found for {matri_id}")

        return result

    except Exception as e:
        logger.error(f"[Rekognition] Error searching face: {e}")
        result["error"] = str(e)
        return result


def find_primary_in_rekognition_by_matri_id(matri_id: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Check if matri_id has an indexed face in Rekognition collections.

    This searches by ExternalImageId which contains the matri_id.

    Returns:
        (found, face_id, s3_key)
    """
    try:
        rekognition = get_rekognition_client()
        if rekognition is None:
            return False, None, None

        # List faces in collections and find ones matching this matri_id
        for collection_id in [REKOGNITION_COLLECTION_1, REKOGNITION_COLLECTION_2]:
            try:
                # Use list_faces with ExternalImageId doesn't work directly,
                # so we need to search by listing faces
                paginator = rekognition.get_paginator('list_faces')

                for page in paginator.paginate(CollectionId=collection_id, MaxResults=100):
                    for face in page.get('Faces', []):
                        external_id = face.get('ExternalImageId', '')
                        # Check if this face belongs to the matri_id
                        if external_id.startswith(matri_id):
                            logger.info(f"[Rekognition] Found indexed face for {matri_id}: {external_id}")
                            return True, face['FaceId'], external_id

            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ResourceNotFoundException':
                    logger.info(f"[Rekognition] Collection {collection_id} not found")
                else:
                    logger.error(f"[Rekognition] Error listing faces in {collection_id}: {e}")
                continue

        logger.info(f"[Rekognition] No indexed face found for {matri_id}")
        return False, None, None

    except Exception as e:
        logger.error(f"[Rekognition] Error finding primary: {e}")
        return False, None, None


def match_face_against_collection(image_path: str, matri_id: str) -> dict:
    """
    Use Rekognition search_faces_by_image to check if the face in image_path
    matches an indexed face for the given matri_id in the collections.

    This is used for existing users (primary already indexed) to verify
    secondary photos match the primary person - similar to the Lambda
    code's psmatch==2 approach.

    Returns:
        dict with keys: matched, similarity, face_id, external_id, details
    """
    result = {
        "matched": False,
        "similarity": 0.0,
        "face_id": None,
        "external_id": None,
        "details": None,
        "error": None
    }

    try:
        rekognition = get_rekognition_client()
        if rekognition is None:
            result["error"] = "Rekognition client not available"
            return result

        # Read image bytes (convert WebP/AVIF/HEIF to JPEG if needed)
        image_bytes = get_image_bytes_for_rekognition(image_path)

        all_matches = []

        for collection_id in [REKOGNITION_COLLECTION_1, REKOGNITION_COLLECTION_2]:
            try:
                response = rekognition.search_faces_by_image(
                    CollectionId=collection_id,
                    Image={'Bytes': image_bytes},
                    MaxFaces=10,
                    FaceMatchThreshold=99.0
                )

                if 'FaceMatches' in response:
                    all_matches.extend(response['FaceMatches'])

            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'InvalidParameterException':
                    # No face detected in the image
                    logger.info(f"[Rekognition] No face detected in image for search: {e}")
                    result["error"] = "No face detected in image"
                    return result
                elif error_code == 'ResourceNotFoundException':
                    logger.info(f"[Rekognition] Collection {collection_id} not found")
                    continue
                else:
                    logger.error(f"[Rekognition] Error searching in {collection_id}: {e}")
                    continue

        # Check if any match belongs to this matri_id with similarity >= 99%
        best_match = None
        best_similarity = 0.0

        for face_match in all_matches:
            similarity = face_match['Similarity']
            external_id = face_match['Face'].get('ExternalImageId', '')

            # Check if this face belongs to the matri_id
            if external_id.startswith(matri_id) or external_id == matri_id:
                if similarity >= 99.0 and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = face_match

        if best_match:
            result["matched"] = True
            result["similarity"] = best_similarity
            result["face_id"] = best_match['Face']['FaceId']
            result["external_id"] = best_match['Face'].get('ExternalImageId', '')
            result["details"] = f"Face matched in Rekognition collection (similarity: {best_similarity:.2f}%)"
            logger.info(f"[Rekognition] Face match found for {matri_id}: similarity={best_similarity:.2f}%, external_id={result['external_id']}")
        else:
            result["details"] = f"No matching face found for {matri_id} in collections"
            logger.info(f"[Rekognition] No face match found for {matri_id} in collections")

        return result

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[Rekognition] Error in face match search: {e}")
        return result


def find_primary_photo_in_s3(matri_id: str, external_id: str = None) -> Optional[str]:
    """
    Search for an existing approved primary photo for the matri_id in S3.

    If external_id is provided (from Rekognition), use it directly.
    Otherwise search by naming convention.

    Returns the S3 key if found, None otherwise.
    """
    try:
        s3_client = get_s3_client()
        if s3_client is None:
            logger.warning(f"[S3] S3 client not available, skipping primary photo lookup")
            return None

        # If we have external_id from Rekognition, construct S3 path
        if external_id:
            # External ID is typically the matri_id, need to find the actual file
            # Search with the external_id as prefix
            prefixes_to_search = [
                f"input/{external_id}",
                f"crop/apitest/{external_id}",
            ]
        else:
            prefixes_to_search = [
                f"input/{matri_id}_",
                f"crop/apitest/{matri_id}_",
            ]

        for prefix in prefixes_to_search:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=S3_BUCKET_NAME,
                    Prefix=prefix,
                    MaxKeys=10
                )

                if 'Contents' in response and len(response['Contents']) > 0:
                    for obj in response['Contents']:
                        key = obj['Key']
                        if '_primary' in key.lower() or key.endswith(('.jpg', '.jpeg', '.png')):
                            logger.info(f"[S3] Found primary photo for {matri_id}: {key}")
                            return key
            except ClientError as e:
                logger.error(f"[S3] Error searching prefix {prefix}: {e}")
                continue

        # Search with date-based paths
        today = datetime.utcnow()
        for days_ago in range(7):
            from datetime import timedelta
            search_date = today - timedelta(days=days_ago)
            date_prefix = search_date.strftime("%Y-%m-%d")

            for subfolder in ["input", "crop/apitest"]:
                prefix = f"{date_prefix}/{subfolder}/{matri_id}_"
                try:
                    response = s3_client.list_objects_v2(
                        Bucket=S3_BUCKET_NAME,
                        Prefix=prefix,
                        MaxKeys=5
                    )

                    if 'Contents' in response and len(response['Contents']) > 0:
                        for obj in response['Contents']:
                            key = obj['Key']
                            if '_primary' in key.lower() or key.endswith(('.jpg', '.jpeg', '.png')):
                                logger.info(f"[S3] Found primary photo for {matri_id}: {key}")
                                return key
                except ClientError:
                    continue

        logger.info(f"[S3] No existing primary photo found for {matri_id}")
        return None

    except Exception as e:
        logger.error(f"[S3] Error searching for primary photo: {e}")
        return None


def download_primary_from_s3(s3_key: str, matri_id: str) -> Optional[str]:
    """
    Download primary photo from S3 to a temporary file.

    Returns the local temp file path if successful, None otherwise.
    """
    try:
        s3_client = get_s3_client()
        if s3_client is None:
            return None

        # Create temp directory if not exists
        temp_dir = os.path.join(tempfile.gettempdir(), "photo_uploads", "s3_primary")
        os.makedirs(temp_dir, exist_ok=True)

        # Get file extension from S3 key
        file_extension = os.path.splitext(s3_key)[1] or ".jpg"
        temp_path = os.path.join(temp_dir, f"{matri_id}_primary_{uuid.uuid4()}{file_extension}")

        # Download file
        s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_path)
        logger.info(f"[S3] Downloaded primary photo to: {temp_path}")

        return temp_path

    except ClientError as e:
        logger.error(f"[S3] Error downloading primary photo: {e}")
        return None
    except Exception as e:
        logger.error(f"[S3] Unexpected error downloading: {e}")
        return None


def index_face_to_collection(image_path: str, matri_id: str, s3_key: str = None) -> dict:
    """
    Index a face to Rekognition collection for future duplicate detection.

    This should be called after a primary photo is approved.

    Args:
        image_path: Local path to the image
        matri_id: The user's matri_id
        s3_key: Optional S3 key where the image is stored

    Returns:
        dict with face_id, success status, and any errors
    """
    result = {
        "success": False,
        "face_id": None,
        "faces_indexed": 0,
        "faces_unindexed": 0,
        "error": None,
        "uat_mode": UAT_MODE,
        "skip_aws": SKIP_AWS_CHECKS
    }

    # Skip if AWS checks are disabled (no credentials)
    if SKIP_AWS_CHECKS:
        result["success"] = True
        result["face_id"] = "AWS_CHECKS_SKIPPED"
        result["error"] = None
        logger.warning(f"[SKIP AWS] Skipping face indexing for {matri_id} - AWS checks disabled")
        return result

    # Skip indexing in UAT mode
    if UAT_MODE:
        result["success"] = True
        result["face_id"] = "UAT_MODE_SKIPPED"
        result["error"] = None
        logger.warning(f"[UAT MODE] Skipping face indexing for {matri_id} - write operations disabled")
        return result

    try:
        rekognition = get_rekognition_client()
        if rekognition is None:
            result["error"] = "Rekognition client not available"
            return result

        # Read image bytes (convert WebP/AVIF/HEIF to JPEG if needed)
        image_bytes = get_image_bytes_for_rekognition(image_path)

        # Use REKOGNITION_COLLECTION_2 for new indexing (as per original workflow)
        response = rekognition.index_faces(
            CollectionId=REKOGNITION_COLLECTION_2,
            Image={'Bytes': image_bytes},
            ExternalImageId=matri_id,  # Use matri_id as external ID
            MaxFaces=1,
            QualityFilter="AUTO",
            DetectionAttributes=['ALL']
        )

        faces_indexed = len(response.get('FaceRecords', []))
        faces_unindexed = len(response.get('UnindexedFaces', []))

        result["faces_indexed"] = faces_indexed
        result["faces_unindexed"] = faces_unindexed

        if faces_indexed > 0:
            result["success"] = True
            result["face_id"] = response['FaceRecords'][0]['Face']['FaceId']
            logger.info(f"[Rekognition] Indexed face for {matri_id}: {result['face_id']}")
        else:
            reasons = [uf.get('Reasons', []) for uf in response.get('UnindexedFaces', [])]
            result["error"] = f"Face not indexed. Reasons: {reasons}"
            logger.error(f"[Rekognition] Failed to index face for {matri_id}: {reasons}")

        return result

    except ClientError as e:
        error_code = e.response['Error']['Code']
        result["error"] = f"{error_code}: {e.response['Error']['Message']}"
        logger.error(f"[Rekognition] Error indexing face: {e}")
        return result
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[Rekognition] Unexpected error indexing face: {e}")
        return result


def check_existing_primary_by_face_search(image_path: str, matri_id: str) -> dict:
    """
    Check if the uploaded face already exists in Rekognition collections
    using search_faces_by_image (fast lookup, no pagination).

    Returns:
        dict with:
            - has_existing_primary: bool - True if same matri_id found in collection
            - is_duplicate: bool - True if different matri_id matched (duplicate)
            - face_id: str - matched face ID
            - matched_matri_id: str - the matri_id that matched (for duplicates)
            - similarity: float - match similarity
    """
    result = {
        "has_existing_primary": False,
        "is_duplicate": False,
        "face_id": None,
        "matched_matri_id": None,
        "similarity": 0.0,
        "error": None
    }

    if SKIP_AWS_CHECKS:
        logger.warning(f"[SKIP AWS] Skipping existing primary check for {matri_id} - AWS checks disabled")
        return result

    try:
        rekognition = get_rekognition_client()
        if rekognition is None:
            result["error"] = "Rekognition client not available"
            return result

        # Read image bytes (convert WebP/AVIF/HEIF to JPEG if needed)
        image_bytes = get_image_bytes_for_rekognition(image_path)

        all_matches = []

        for collection_id in [REKOGNITION_COLLECTION_1, REKOGNITION_COLLECTION_2]:
            try:
                response = rekognition.search_faces_by_image(
                    CollectionId=collection_id,
                    Image={'Bytes': image_bytes},
                    MaxFaces=10,
                    FaceMatchThreshold=FACE_MATCH_THRESHOLD
                )
                if 'FaceMatches' in response:
                    all_matches.extend(response['FaceMatches'])

            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'InvalidParameterException':
                    logger.info(f"[Rekognition] No face detected in image for existing primary check")
                elif error_code == 'ResourceNotFoundException':
                    logger.info(f"[Rekognition] Collection {collection_id} not found")
                else:
                    logger.error(f"[Rekognition] Error searching {collection_id}: {e}")
                continue

        # Process matches - find best match
        for match in all_matches:
            similarity = match['Similarity']
            external_id = match['Face'].get('ExternalImageId', '')
            face_id = match['Face']['FaceId']
            matched_id = external_id.partition("_")[0] if external_id else ""

            if similarity >= 99.0:
                if matched_id == matri_id:
                    # Same matri_id - existing user
                    if similarity > result["similarity"] or not result["has_existing_primary"]:
                        result["has_existing_primary"] = True
                        result["face_id"] = face_id
                        result["matched_matri_id"] = matched_id
                        result["similarity"] = similarity
                        logger.info(f"[Rekognition] Existing primary found for {matri_id} (similarity: {similarity:.2f}%)")
                else:
                    # Different matri_id - duplicate
                    if not result["is_duplicate"] or similarity > result["similarity"]:
                        result["is_duplicate"] = True
                        result["face_id"] = face_id
                        result["matched_matri_id"] = matched_id
                        result["similarity"] = similarity
                        logger.info(f"[Rekognition] DUPLICATE: {matri_id} matches {matched_id} (similarity: {similarity:.2f}%)")

        if not result["has_existing_primary"] and not result["is_duplicate"]:
            logger.info(f"[Rekognition] No existing primary found for {matri_id} - new user")

        return result

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[Rekognition] Error in existing primary check: {e}")
        return result


def load_image_for_detection(image_path: str):
    """Load image for RetinaFace detection, handling various formats"""
    try:
        pil_img = Image.open(image_path)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def detect_face_count(image_path: str) -> int:
    """Detect number of faces in an image using RetinaFace"""
    try:
        img = load_image_for_detection(image_path)
        if img is None:
            return 0
        faces = RetinaFace.detect_faces(img)
        if not faces or isinstance(faces, tuple):
            return 0
        return len(faces)
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return 0

def is_individual_photo(image_path: str) -> bool:
    """Check if photo contains exactly one face (individual photo)"""
    return detect_face_count(image_path) == 1

def get_gpu_info() -> dict:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return {
            "available": GPU_AVAILABLE,
            "tensorflow_gpu": TF_GPU_AVAILABLE,
            "onnx_gpu": ONNX_GPU_AVAILABLE,
            "nudenet_gpu": ONNX_GPU_AVAILABLE,
            "gpu_count": len(gpus),
            "cuda_available": tf.test.is_built_with_cuda(),
        }
    except:
        return {
            "available": GPU_AVAILABLE,
            "tensorflow_gpu": TF_GPU_AVAILABLE,
            "onnx_gpu": ONNX_GPU_AVAILABLE,
            "nudenet_gpu": ONNX_GPU_AVAILABLE,
        }

def validate_single_image_sync(temp_path, photo_type, profile_data, reference_path=None, use_deepface_gender=False, rekognition_face_match=None):
    try:
        result = validate_photo_complete_hybrid(
            image_path=temp_path,
            photo_type=photo_type,
            profile_data=profile_data,
            reference_photo_path=reference_path,
            run_stage2=True,
            use_deepface_gender=use_deepface_gender,
            rekognition_face_match=rekognition_face_match
        )
        return result
    except Exception as e:
        return {
            "final_decision": "ERROR",
            "final_action": "ERROR",
            "final_reason": str(e),
            "stage1": {"checks": {}},
        }

# ==================== SINGLE IMAGE ENDPOINTS ====================

# @app.post("/api/v3/validate/single/primary", response_model=SingleImageResponse)
# async def validate_single_primary_photo(
#     photo: UploadFile = File(...),
#     matri_id: str = Form(...),
#     gender: str = Form(...),
#     age: int = Form(...),
#     use_deepface_gender: bool = Form(False)
# ):
#     """Validate single PRIMARY photo with full GPU acceleration"""
#     start_time = time.time()
#     temp_file_path = None
    
#     try:
#         if age < 18:
#             raise HTTPException(status_code=400, detail="Age must be 18+")
#         if gender not in ["Male", "Female"]:
#             raise HTTPException(status_code=400, detail="Gender must be Male/Female")
        
#         temp_file_path = save_upload_file_tmp(photo)
        
#         profile_data = {"matri_id": matri_id, "gender": gender, "age": age}
        
#         result = validate_photo_complete_hybrid(
#             image_path=temp_file_path,
#             photo_type="PRIMARY",
#             profile_data=profile_data,
#             run_stage2=True,
#             use_deepface_gender=use_deepface_gender
#         )
        
#         cleanup_temp_files(temp_file_path)
        
#         response_data = format_validation_result(result, photo.filename)
#         response_time = round(time.time() - start_time, 3)
        
#         return SingleImageResponse(
#             success=response_data["final_status"] == "ACCEPTED",
#             message=response_data["final_reason"],
#             data=response_data,
#             response_time_seconds=response_time,
#             library_usage=response_data.get("library_usage"),
#             gpu_info=get_gpu_info()
#         )
#     except Exception as e:
#         cleanup_temp_files(temp_file_path)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/v3/validate/single/secondary", response_model=SingleImageResponse)
# async def validate_single_secondary_photo(
#     photo: UploadFile = File(...),
#     matri_id: str = Form(...),
#     gender: str = Form(...),
#     age: int = Form(...),
#     reference_photo: Optional[UploadFile] = File(None)
# ):
#     """Validate single SECONDARY photo"""
#     start_time = time.time()
#     temp_file_path = None
#     temp_reference_path = None
    
#     try:
#         if age < 18:
#             raise HTTPException(status_code=400, detail="Age must be 18+")
        
#         temp_file_path = save_upload_file_tmp(photo)
        
#         if reference_photo:
#             temp_reference_path = save_upload_file_tmp(reference_photo)
        
#         profile_data = {"matri_id": matri_id, "gender": gender, "age": age}
        
#         result = validate_photo_complete_hybrid(
#             image_path=temp_file_path,
#             photo_type="SECONDARY",
#             profile_data=profile_data,
#             reference_photo_path=temp_reference_path,
#             run_stage2=True
#         )
        
#         cleanup_temp_files(temp_file_path, temp_reference_path)
        
#         response_data = format_validation_result(result, photo.filename)
#         response_time = round(time.time() - start_time, 3)
        
#         return SingleImageResponse(
#             success=response_data["final_status"] == "ACCEPTED",
#             message=response_data["final_reason"],
#             data=response_data,
#             response_time_seconds=response_time,
#             library_usage=response_data.get("library_usage"),
#             gpu_info=get_gpu_info()
#         )
#     except Exception as e:
#         cleanup_temp_files(temp_file_path, temp_reference_path)
#         raise HTTPException(status_code=500, detail=str(e))

# # ==================== BATCH ENDPOINTS (FIXED FOR SWAGGER UI) ====================

# @app.post("/api/v3/validate/batch/primary", response_model=MultiImageResponse)
# async def validate_batch_primary_photos(
#     matri_id: str = Form(...),
#     gender: str = Form(...),
#     age: int = Form(...),
#     use_deepface_gender: bool = Form(False),
#     photos: List[UploadFile] = File(...)  # FIXED: List must be last parameter
# ):
#     """
#     Validate multiple PRIMARY photos in batch
    
#     **FIXED**: Now works in Swagger UI
#     **Note**: Upload multiple files by clicking the file input multiple times
#     """
#     start_time = time.time()
#     temp_files = []
    
#     try:
#         if len(photos) > 10:
#             raise HTTPException(status_code=400, detail="Max 10 images per batch")
#         if age < 18:
#             raise HTTPException(status_code=400, detail="Age must be 18+")
        
#         profile_data = {"matri_id": matri_id, "gender": gender, "age": age}
        
#         for photo in photos:
#             temp_path = save_upload_file_tmp(photo)
#             temp_files.append((temp_path, photo.filename))
        
#         loop = asyncio.get_event_loop()
#         validation_tasks = []
        
#         for temp_path, filename in temp_files:
#             task = loop.run_in_executor(
#                 executor,
#                 validate_single_image_sync,
#                 temp_path,
#                 "PRIMARY",
#                 profile_data,
#                 None,
#                 use_deepface_gender
#             )
#             validation_tasks.append((task, filename))
        
#         results = []
#         for task, filename in validation_tasks:
#             result = await task
#             formatted_result = format_validation_result(result, filename)
#             results.append(formatted_result)
        
#         cleanup_temp_files(*[path for path, _ in temp_files])
        
#         response_time = round(time.time() - start_time, 3)
        
#         results = convert_numpy_types(results)
        
#         summary = {
#             "total": len(results),
#             "approved": sum(1 for r in results if r["final_decision"] == "APPROVE"),
#             "rejected": sum(1 for r in results if r["final_decision"] == "REJECT"),
#             "suspended": sum(1 for r in results if r["final_decision"] == "SUSPEND"),
#             "review_needed": sum(1 for r in results if r["final_decision"] == "MANUAL_REVIEW"),
#             "processing_time_seconds": response_time,
#             "avg_time_per_image": round(response_time / len(results), 3) if results else 0
#         }
        
#         library_usage_summary = {
#             "insightface_used": sum(1 for r in results if r.get("library_usage")),
#             "deepface_used": sum(1 for r in results if r.get("library_usage") and r["library_usage"].get("deepface")),
#             "nudenet_gpu": ONNX_GPU_AVAILABLE,
#             "gpu_accelerated_count": sum(1 for r in results if (r.get("library_usage") or {}).get("gpu_used", False))
#         }
        
#         return MultiImageResponse(
#             success=True,
#             message=f"Batch validation: {summary['approved']} approved, {summary['rejected']} rejected",
#             batch_id=str(uuid.uuid4()),
#             total_images=len(results),
#             results=results,
#             summary=summary,
#             response_time_seconds=response_time,
#             library_usage_summary=library_usage_summary,
#             gpu_info=get_gpu_info()
#         )
#     except Exception as e:
#         cleanup_temp_files(*[path for path, _ in temp_files])
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/v3/validate/batch/secondary", response_model=MultiImageResponse)
# async def validate_batch_secondary_photos(
#     matri_id: str = Form(...),
#     gender: str = Form(...),
#     age: int = Form(...),
#     reference_photo: Optional[UploadFile] = File(None),
#     photos: List[UploadFile] = File(...)  # FIXED: List must be last
# ):
#     """
#     Validate multiple SECONDARY photos in batch
    
#     **FIXED**: Now works in Swagger UI
#     """
#     start_time = time.time()
#     temp_files = []
#     temp_reference_path = None
    
#     try:
#         if len(photos) > 10:
#             raise HTTPException(status_code=400, detail="Max 10 images")
#         if age < 18:
#             raise HTTPException(status_code=400, detail="Age must be 18+")
        
#         profile_data = {"matri_id": matri_id, "gender": gender, "age": age}
        
#         if reference_photo:
#             temp_reference_path = save_upload_file_tmp(reference_photo)
        
#         for photo in photos:
#             temp_path = save_upload_file_tmp(photo)
#             temp_files.append((temp_path, photo.filename))
        
#         loop = asyncio.get_event_loop()
#         validation_tasks = []
        
#         for temp_path, filename in temp_files:
#             task = loop.run_in_executor(
#                 executor,
#                 validate_single_image_sync,
#                 temp_path,
#                 "SECONDARY",
#                 profile_data,
#                 temp_reference_path,
#                 False
#             )
#             validation_tasks.append((task, filename))
        
#         results = []
#         for task, filename in validation_tasks:
#             result = await task
#             formatted_result = format_validation_result(result, filename)
#             results.append(formatted_result)
        
#         cleanup_temp_files(*[path for path, _ in temp_files], temp_reference_path)
        
#         response_time = round(time.time() - start_time, 3)
#         results = convert_numpy_types(results)
        
#         summary = {
#             "total": len(results),
#             "approved": sum(1 for r in results if r["final_decision"] == "APPROVE"),
#             "rejected": sum(1 for r in results if r["final_decision"] == "REJECT"),
#             "suspended": sum(1 for r in results if r["final_decision"] == "SUSPEND"),
#             "review_needed": sum(1 for r in results if r["final_decision"] == "MANUAL_REVIEW"),
#             "processing_time_seconds": response_time,
#             "avg_time_per_image": round(response_time / len(results), 3) if results else 0
#         }
        
#         return MultiImageResponse(
#             success=True,
#             message=f"Batch validation: {summary['approved']} approved",
#             batch_id=str(uuid.uuid4()),
#             total_images=len(results),
#             results=results,
#             summary=summary,
#             response_time_seconds=response_time,
#             gpu_info=get_gpu_info()
#         )
#     except Exception as e:
#         cleanup_temp_files(*[path for path, _ in temp_files], temp_reference_path)
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/validatephoto")
async def validate_photo_auto_detect(
    matri_id: str = Form(...),
    gender: str = Form(...),
    age: int = Form(...),
    product_name: str = Form(...),
    use_deepface_gender: bool = Form(False),
    Photo_upload: List[UploadFile] = File(...)
):
    """
    Validate photos with automatic PRIMARY/SECONDARY detection.

    **For Existing Users (primary already approved in S3):**
    - System checks if matri_id already has an approved primary photo in S3
    - If found, all uploaded photos are validated as SECONDARY against S3 primary

    **For New Users (no existing primary):**
    - All photos are uploaded via single 'Photo_upload' parameter
    - Individual photos (single face) can be PRIMARY
    - Group photos (multiple faces) are always SECONDARY
    - System automatically finds a valid PRIMARY photo first, then validates others as SECONDARY
    - If no individual photo found or none pass validation, request is rejected
    """
    start_time = time.time()
    temp_files = []  # List of (temp_path, filename, original_index)
    primary_photo_path = None
    s3_primary_path = None  # Track S3 downloaded primary for cleanup
    existing_primary_used = False  # Flag to track if we used S3 primary

    try:
        total_photos = len(Photo_upload) if Photo_upload else 0

        if total_photos == 0:
            raise HTTPException(status_code=400, detail="At least one photo required")
        if total_photos > 10:
            raise HTTPException(status_code=400, detail="Max 10 images")
        if age < 18:
            raise HTTPException(status_code=400, detail="Age must be 18+")

        profile_data = {"matri_id": matri_id, "gender": gender, "age": age, "product_name": product_name}
        batch_id = str(uuid.uuid4())
        gpu_info = get_gpu_info()

        # ==================== SAVE FIRST PHOTO FOR EXISTING PRIMARY CHECK ====================
        # Save first photo to use for face search (fast lookup via search_faces_by_image)
        first_photo_temp = save_upload_file_tmp(Photo_upload[0])
        temp_files.append((first_photo_temp, Photo_upload[0].filename, 0))

        # Check if this face already exists in Rekognition collections
        primary_check = check_existing_primary_by_face_search(first_photo_temp, matri_id)

        # If duplicate detected (different matri_id matched at 99%+), reject immediately
        if primary_check.get("is_duplicate"):
            dup_matri_id = primary_check["matched_matri_id"]
            dup_similarity = primary_check["similarity"]
            logger.info(f"[Rekognition] DUPLICATE detected: {matri_id} matches existing {dup_matri_id} ({dup_similarity:.2f}%)")

            cleanup_temp_files(*[path for path, _, _ in temp_files])
            response_time = round(time.time() - start_time, 3)

            return JSONResponse(status_code=409, content=convert_numpy_types({
                "status_code": 409,
                "error_code": "DUPLICATE_DETECTED",
                "success": False,
                "message": f"REJECTED: Duplicate matri_id detected. The uploaded photo matches an existing profile with matri_id: {dup_matri_id} (similarity: {dup_similarity:.2f}%).",
                "product_name": product_name,
                "batch_id": batch_id,
                "total_images": total_photos,
                "duplicate_detected": True,
                "duplicate_matri_id": dup_matri_id,
                "duplicate_similarity": dup_similarity,
                "results": {
                    "primary": [],
                    "secondary": [],
                    "duplicate_match": {
                        "uploaded_matri_id": matri_id,
                        "matched_matri_id": dup_matri_id,
                        "similarity": dup_similarity,
                        "face_id": primary_check["face_id"]
                    }
                },
                "summary": {
                    "total": total_photos,
                    "primary_count": 0,
                    "secondary_count": 0,
                    "approved": 0,
                    "rejected": total_photos,
                    "rejection_reason": f"Duplicate matri_id detected. Photo matches existing profile: {dup_matri_id}"
                },
                "response_time_seconds": response_time,
                "gpu_info": gpu_info
            }))

        has_existing_primary = primary_check.get("has_existing_primary", False)
        existing_face_id = primary_check.get("face_id")

        if has_existing_primary:
            logger.info(f"[Rekognition] Existing user detected for {matri_id}: face_id={existing_face_id}")
            logger.info(f"[Rekognition] Secondary photos will be matched via search_faces_by_image")
            existing_primary_used = True

            # All uploaded photos will be validated as SECONDARY
            loop = asyncio.get_event_loop()
            results = {"primary": [], "secondary": []}

            # Save remaining uploaded photos (first already saved)
            for idx, photo in enumerate(Photo_upload):
                if idx == 0:
                    # First photo already saved
                    temp_path = first_photo_temp
                else:
                    temp_path = save_upload_file_tmp(photo)
                    temp_files.append((temp_path, photo.filename, idx))
                face_count = detect_face_count(temp_path)

                # Determine auto-detected type
                if face_count == 0:
                    auto_type = "NO_FACE"
                elif face_count == 1:
                    auto_type = "INDIVIDUAL"
                else:
                    auto_type = "GROUP"

                if face_count == 0:
                    # No face detected - reject this photo
                    invalid_result = {
                        "image_filename": photo.filename,
                        "validation_id": str(uuid.uuid4()),
                        "timestamp": datetime.utcnow().isoformat(),
                        "photo_type": "SECONDARY",
                        "final_status": "REJECTED",
                        "final_reason": "No face detected in photo",
                        "final_action": "REJECT",
                        "final_decision": "REJECT",
                        "matri_id": matri_id,
                        "original_upload_index": idx,
                        "auto_detected_type": auto_type,
                        "reference_photo": f"rekognition:{existing_face_id}"
                    }
                    results["secondary"].append(invalid_result)
                else:
                    # Match face against Rekognition collection before validation
                    rekognition_match = match_face_against_collection(temp_path, matri_id)

                    # Validate as SECONDARY with Rekognition face match result
                    result = await loop.run_in_executor(
                        executor,
                        validate_single_image_sync,
                        temp_path,
                        "SECONDARY",
                        profile_data,
                        None,  # No local reference photo - using Rekognition
                        False,
                        rekognition_match  # Pass Rekognition match result
                    )

                    formatted_result = format_validation_result(result, photo.filename)
                    formatted_result["matri_id"] = matri_id
                    formatted_result["original_upload_index"] = idx
                    formatted_result["auto_detected_type"] = auto_type
                    formatted_result["reference_photo"] = f"rekognition:{existing_face_id}"
                    formatted_result["existing_primary_used"] = True
                    formatted_result["rekognition_face_match"] = rekognition_match
                    results["secondary"].append(formatted_result)

            # Cleanup temp files
            cleanup_temp_files(*[path for path, _, _ in temp_files])

            response_time = round(time.time() - start_time, 3)
            results = convert_numpy_types(results)

            all_results = results["secondary"]

            # Save all validation results to database
            for validation_result in all_results:
                save_validation_to_db(
                    validation_data=validation_result,
                    matri_id=matri_id,
                    batch_id=batch_id,
                    response_time=response_time / len(all_results) if all_results else response_time,
                    gpu_info=gpu_info,
                    product_name=product_name
                )

            summary = {
                "total": len(all_results),
                "existing_primary_used": True,
                "existing_primary_face_id": existing_face_id,
                "primary_count": 0,
                "secondary_count": len(all_results),
                "approved": sum(1 for r in all_results if r.get("final_decision") == "APPROVE"),
                "rejected": sum(1 for r in all_results if r.get("final_decision") in ["REJECT", "SUSPEND"]),
                "processing_time_seconds": response_time
            }

            return {
                "status_code": 200,
                "success": True,
                "message": f"Existing user validation complete. Primary verified via Rekognition collection. {summary['approved']} secondary photos approved, {summary['rejected']} rejected.",
                "product_name": product_name,
                "batch_id": batch_id,
                "total_images": len(all_results),
                "existing_primary_used": True,
                "existing_primary_face_id": existing_face_id,
                "results": results,
                "summary": summary,
                "response_time_seconds": response_time,
                "gpu_info": gpu_info
            }

        # ==================== NEW USER FLOW (NO EXISTING PRIMARY) ====================
        # Step 1: Save all uploaded files and detect face counts
        photo_analysis = []  # List of (temp_path, filename, face_count, original_index)

        for idx, photo in enumerate(Photo_upload):
            if idx == 0:
                # First photo already saved during existing primary check
                temp_path = first_photo_temp
            else:
                temp_path = save_upload_file_tmp(photo)
                temp_files.append((temp_path, photo.filename, idx))
            face_count = detect_face_count(temp_path)
            photo_analysis.append({
                "temp_path": temp_path,
                "filename": photo.filename,
                "face_count": face_count,
                "original_index": idx,
                "is_individual": face_count == 1,
                "is_group": face_count > 1
            })

        # Step 2: Separate individual and group photos
        individual_photos = [p for p in photo_analysis if p["is_individual"]]
        group_photos = [p for p in photo_analysis if p["is_group"]]
        invalid_photos = [p for p in photo_analysis if p["face_count"] == 0]

        # Step 3: Check if we have any individual photos
        if len(individual_photos) == 0:
            # All photos are group photos or have no faces - reject
            cleanup_temp_files(*[path for path, _, _ in temp_files])

            if len(group_photos) > 0:
                # Build list of group photos with face counts
                group_details = [
                    {"filename": p["filename"], "faces_detected": p["face_count"]}
                    for p in group_photos
                ]

                return JSONResponse(status_code=422, content=convert_numpy_types({
                    "status_code": 422,
                    "error_code": "NO_INDIVIDUAL_PHOTO",
                    "success": False,
                    "message": f"REJECTED: Primary photo not found. All {len(group_photos)} uploaded photo(s) are group photos (containing multiple faces). A primary photo must contain only YOUR face (single person). Please upload at least one individual photo to proceed.",
                    "product_name": product_name,
                    "batch_id": batch_id,
                    "total_images": total_photos,
                    "results": {
                        "primary": [],
                        "secondary": [],
                        "group_photos_detected": group_details
                    },
                    "summary": {
                        "total": total_photos,
                        "primary_count": 0,
                        "secondary_count": 0,
                        "group_photos_found": len(group_photos),
                        "invalid_photos": len(invalid_photos),
                        "approved": 0,
                        "rejected": total_photos,
                        "rejection_reason": "No individual photo found to use as primary. All photos contain multiple faces (group photos).",
                        "group_photo_details": group_details
                    },
                    "response_time_seconds": round(time.time() - start_time, 3),
                    "gpu_info": gpu_info
                }))
            else:
                # Build list of invalid photos
                invalid_details = [
                    {"filename": p["filename"], "faces_detected": 0}
                    for p in invalid_photos
                ]

                return JSONResponse(status_code=422, content=convert_numpy_types({
                    "status_code": 422,
                    "error_code": "NO_FACE_DETECTED",
                    "success": False,
                    "message": f"REJECTED: No valid faces detected in any of the {total_photos} uploaded photo(s). Please ensure photos contain clear, visible faces and try again.",
                    "product_name": product_name,
                    "batch_id": batch_id,
                    "total_images": total_photos,
                    "results": {
                        "primary": [],
                        "secondary": [],
                        "invalid_photos_detected": invalid_details
                    },
                    "summary": {
                        "total": total_photos,
                        "primary_count": 0,
                        "secondary_count": 0,
                        "approved": 0,
                        "rejected": total_photos,
                        "rejection_reason": "No faces detected in any uploaded photos",
                        "invalid_photo_details": invalid_details
                    },
                    "response_time_seconds": round(time.time() - start_time, 3),
                    "gpu_info": gpu_info
                }))

        # Step 4: Try to validate individual photos as PRIMARY until one passes
        loop = asyncio.get_event_loop()
        primary_result = None
        primary_photo_info = None
        failed_primary_attempts = []

        for photo_info in individual_photos:
            # Validate as PRIMARY
            result = await loop.run_in_executor(
                executor,
                validate_single_image_sync,
                photo_info["temp_path"],
                "PRIMARY",
                profile_data,
                None,
                use_deepface_gender
            )

            formatted_result = format_validation_result(result, photo_info["filename"])
            formatted_result["matri_id"] = matri_id
            formatted_result["original_upload_index"] = photo_info["original_index"]
            formatted_result["auto_detected_type"] = "INDIVIDUAL"

            if result["final_decision"] == "APPROVE":
                # Found a valid primary photo - now check for duplicates in collection
                duplicate_check = search_face_in_rekognition(photo_info["temp_path"], matri_id)

                if duplicate_check.get("diff_id_matches"):
                    # Face matches a DIFFERENT matri_id with >= 99% similarity - DUPLICATE
                    best_duplicate = max(duplicate_check["diff_id_matches"], key=lambda x: x["similarity"])
                    duplicate_matri_id = best_duplicate["matched_matri_id"]
                    duplicate_similarity = best_duplicate["similarity"]

                    logger.info(f"[Rekognition] DUPLICATE detected: {matri_id} matches existing {duplicate_matri_id} ({duplicate_similarity:.2f}%)")

                    # Cleanup and return duplicate response
                    cleanup_temp_files(*[path for path, _, _ in temp_files])
                    response_time = round(time.time() - start_time, 3)

                    return JSONResponse(status_code=409, content=convert_numpy_types({
                        "status_code": 409,
                        "error_code": "DUPLICATE_DETECTED",
                        "success": False,
                        "message": f"REJECTED: Duplicate matri_id detected. The uploaded photo matches an existing profile with matri_id: {duplicate_matri_id} (similarity: {duplicate_similarity:.2f}%).",
                        "product_name": product_name,
                        "batch_id": batch_id,
                        "total_images": total_photos,
                        "duplicate_detected": True,
                        "duplicate_matri_id": duplicate_matri_id,
                        "duplicate_similarity": duplicate_similarity,
                        "results": {
                            "primary": [],
                            "secondary": [],
                            "duplicate_match": {
                                "uploaded_matri_id": matri_id,
                                "matched_matri_id": duplicate_matri_id,
                                "similarity": duplicate_similarity,
                                "face_id": best_duplicate["face_id"],
                                "external_id": best_duplicate["external_id"]
                            }
                        },
                        "summary": {
                            "total": total_photos,
                            "primary_count": 0,
                            "secondary_count": 0,
                            "approved": 0,
                            "rejected": total_photos,
                            "rejection_reason": f"Duplicate matri_id detected. Photo matches existing profile: {duplicate_matri_id}"
                        },
                        "response_time_seconds": response_time,
                        "gpu_info": gpu_info
                    }))

                # Check if this matri_id already has a face in collection (existing primary photo)
                if duplicate_check.get("same_id_matches"):
                    best_same = max(duplicate_check["same_id_matches"], key=lambda x: x["similarity"])
                    logger.info(f"[Rekognition] Existing primary found for {matri_id} (similarity: {best_same['similarity']:.2f}%)")
                    formatted_result["existing_primary_in_collection"] = True
                    formatted_result["existing_primary_similarity"] = best_same["similarity"]

                # No duplicate - proceed with indexing
                primary_result = formatted_result
                primary_photo_info = photo_info
                primary_photo_path = photo_info["temp_path"]

                # Index the approved primary photo to Rekognition collection
                index_result = index_face_to_collection(
                    image_path=primary_photo_path,
                    matri_id=matri_id
                )
                if index_result["success"]:
                    primary_result["rekognition_indexed"] = True
                    primary_result["rekognition_face_id"] = index_result["face_id"]
                    logger.info(f"[Rekognition] Successfully indexed primary for {matri_id}")
                else:
                    primary_result["rekognition_indexed"] = False
                    primary_result["rekognition_index_error"] = index_result.get("error")
                    logger.error(f"[Rekognition] Failed to index primary for {matri_id}: {index_result.get('error')}")

                break
            else:
                # This individual photo failed primary validation
                failed_primary_attempts.append(formatted_result)

        # Step 5: Check if we found a valid primary photo
        if primary_result is None:
            # No individual photo passed primary validation
            cleanup_temp_files(*[path for path, _, _ in temp_files])

            # Build detailed rejection reasons from failed attempts
            rejection_details = []
            for attempt in failed_primary_attempts:
                reason = attempt.get("final_reason", "Unknown reason")
                filename = attempt.get("image_filename", "Unknown")
                rejection_details.append({
                    "filename": filename,
                    "reason": reason
                })

            # Create a summary message with reasons
            if len(rejection_details) == 1:
                detail_msg = f"Photo '{rejection_details[0]['filename']}' was rejected: {rejection_details[0]['reason']}"
            else:
                detail_msg = "All individual photos failed validation. Reasons: " + "; ".join(
                    [f"'{d['filename']}': {d['reason']}" for d in rejection_details[:3]]  # Show first 3
                )
                if len(rejection_details) > 3:
                    detail_msg += f" (and {len(rejection_details) - 3} more)"

            return JSONResponse(status_code=422, content=convert_numpy_types({
                "status_code": 422,
                "error_code": "PRIMARY_VALIDATION_FAILED",
                "success": False,
                "message": f"REJECTED: Primary photo validation failed. {detail_msg}. Please upload a clear individual photo that meets all requirements.",
                "product_name": product_name,
                "batch_id": batch_id,
                "total_images": total_photos,
                "results": {
                    "primary": [],
                    "secondary": [],
                    "failed_primary_attempts": failed_primary_attempts
                },
                "summary": {
                    "total": total_photos,
                    "individual_photos_found": len(individual_photos),
                    "group_photos_found": len(group_photos),
                    "primary_count": 0,
                    "secondary_count": 0,
                    "approved": 0,
                    "rejected": total_photos,
                    "rejection_reason": "No individual photo passed primary validation",
                    "rejection_details": rejection_details
                },
                "response_time_seconds": round(time.time() - start_time, 3),
                "gpu_info": gpu_info
            }))

        # Step 6: We have a valid primary photo - now validate remaining photos as SECONDARY
        results = {"primary": [primary_result], "secondary": []}

        # Remaining individual photos (those that weren't selected as primary)
        remaining_individual = [p for p in individual_photos if p["temp_path"] != primary_photo_path]

        # Add failed primary attempts as secondary (they might pass as secondary)
        # But skip them since they already failed - user should know about them

        # Validate remaining individual photos as SECONDARY (with face matching)
        for photo_info in remaining_individual:
            if photo_info["temp_path"] == primary_photo_path:
                continue  # Skip the primary photo

            result = await loop.run_in_executor(
                executor,
                validate_single_image_sync,
                photo_info["temp_path"],
                "SECONDARY",
                profile_data,
                primary_photo_path,  # Use primary as reference
                False
            )

            formatted_result = format_validation_result(result, photo_info["filename"])
            formatted_result["matri_id"] = matri_id
            formatted_result["original_upload_index"] = photo_info["original_index"]
            formatted_result["auto_detected_type"] = "INDIVIDUAL"
            formatted_result["reference_photo"] = primary_photo_info["filename"]
            results["secondary"].append(formatted_result)

        # Validate group photos as SECONDARY (with face matching against primary)
        for photo_info in group_photos:
            result = await loop.run_in_executor(
                executor,
                validate_single_image_sync,
                photo_info["temp_path"],
                "SECONDARY",
                profile_data,
                primary_photo_path,  # Use primary as reference
                False
            )

            formatted_result = format_validation_result(result, photo_info["filename"])
            formatted_result["matri_id"] = matri_id
            formatted_result["original_upload_index"] = photo_info["original_index"]
            formatted_result["auto_detected_type"] = "GROUP"
            formatted_result["reference_photo"] = primary_photo_info["filename"]
            results["secondary"].append(formatted_result)

        # Handle photos with no faces detected
        for photo_info in invalid_photos:
            invalid_result = {
                "image_filename": photo_info["filename"],
                "validation_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "photo_type": "INVALID",
                "final_status": "REJECTED",
                "final_reason": "No face detected in photo",
                "final_action": "REJECT",
                "final_decision": "REJECT",
                "matri_id": matri_id,
                "original_upload_index": photo_info["original_index"],
                "auto_detected_type": "NO_FACE"
            }
            results["secondary"].append(invalid_result)

        # Cleanup temp files
        cleanup_temp_files(*[path for path, _, _ in temp_files])

        response_time = round(time.time() - start_time, 3)
        results = convert_numpy_types(results)

        all_results = results["primary"] + results["secondary"]

        # Save all validation results to database
        for validation_result in all_results:
            save_validation_to_db(
                validation_data=validation_result,
                matri_id=matri_id,
                batch_id=batch_id,
                response_time=response_time / len(all_results) if all_results else response_time,
                gpu_info=gpu_info
            )

        summary = {
            "total": len(all_results),
            "individual_photos_found": len(individual_photos),
            "group_photos_found": len(group_photos),
            "invalid_photos": len(invalid_photos),
            "primary_count": len(results["primary"]),
            "secondary_count": len(results["secondary"]),
            "approved": sum(1 for r in all_results if r.get("final_decision") == "APPROVE"),
            "rejected": sum(1 for r in all_results if r.get("final_decision") in ["REJECT", "SUSPEND"]),
            "processing_time_seconds": response_time,
            "primary_photo_used": primary_photo_info["filename"] if primary_photo_info else None
        }

        return {
            "status_code": 200,
            "success": True,
            "message": f"Photo validation complete: {summary['approved']} approved, {summary['rejected']} rejected. Primary photo: {primary_photo_info['filename']}",
            "product_name": product_name,
            "batch_id": batch_id,
            "total_images": len(all_results),
            "results": results,
            "summary": summary,
            "response_time_seconds": response_time,
            "gpu_info": gpu_info
        }

    except HTTPException:
        cleanup_temp_files(*[path for path, _, _ in temp_files])
        raise
    except Exception as e:
        cleanup_temp_files(*[path for path, _, _ in temp_files])
        raise HTTPException(status_code=500, detail=str(e))

# ==================== IMAGE PROCESSING ENDPOINT ====================

@app.post("/api/v1/processphoto")
async def process_photo(
    matri_id: str = Form(...),
    photo: UploadFile = File(...)
):
    """
    Process a single photo with RealESRGAN and upload 14 image variants to S3.

    This is a standalone endpoint — no validation logic, no database save.
    Generates 7 sizes × 2 formats (JPG + WebP) = 14 images.

    Sizes: 75x75, 150x150, 250x250, 300x300, 323x323, 1080x1080, 1080x1780

    S3 path: approved/{matri_id}/{base_name}_{WxH}.{jpg|webp}
    """
    start_time = time.time()
    temp_path = None

    try:
        # Save uploaded file to temp
        temp_path = save_upload_file_tmp(photo)
        base_name = os.path.splitext(photo.filename)[0]

        # Process with RealESRGAN (generates 14 images)
        logger.info(f"[ProcessPhoto] Processing image for {matri_id} with RealESRGAN...")
        process_result = process_image_for_sizes(temp_path, base_name=base_name)

        if not process_result["success"]:
            cleanup_temp_files(temp_path)
            raise HTTPException(status_code=500, detail=f"Image processing failed: {process_result['error']}")

        # Upload each processed image to S3
        uploaded_images = []
        for img_info in process_result["images"]:
            s3_key = f"approved/{matri_id}/{img_info['filename']}"
            upload_result = upload_file_to_s3(img_info["path"], s3_key)

            uploaded_images.append({
                "s3_key": s3_key,
                "filename": img_info["filename"],
                "size_label": img_info["size_label"],
                "format": img_info["format"],
                "file_size_kb": img_info["file_size_kb"],
                "uploaded": upload_result["success"],
            })

            if not upload_result["success"]:
                logger.error(f"[ProcessPhoto] Failed to upload {s3_key}: {upload_result['error']}")

        # Cleanup temporary files
        cleanup_processed_images(process_result["output_dir"])
        cleanup_temp_files(temp_path)

        uploaded_count = sum(1 for img in uploaded_images if img["uploaded"])
        response_time = round(time.time() - start_time, 3)

        return {
            "status_code": 200,
            "success": uploaded_count > 0,
            "message": f"Processed and uploaded {uploaded_count}/{len(uploaded_images)} images for {matri_id}",
            "matri_id": matri_id,
            "original_filename": photo.filename,
            "uploaded_images": uploaded_images,
            "total_uploaded": uploaded_count,
            "total_images": len(uploaded_images),
            "processing_time_seconds": process_result["processing_time"],
            "gpu_used": process_result["gpu_used"],
            "response_time_seconds": response_time,
        }

    except HTTPException:
        if temp_path:
            cleanup_temp_files(temp_path)
        raise
    except Exception as e:
        if temp_path:
            cleanup_temp_files(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

# ==================== POST VALIDATION ENDPOINT ====================

def delete_face_from_collection(face_id: str, collection_id: str = None) -> dict:
    """
    Delete a face from Rekognition collection.

    If collection_id is not provided, tries both collections.

    Args:
        face_id: The Rekognition face ID to delete
        collection_id: Optional specific collection to delete from

    Returns:
        dict with success status and details
    """
    result = {
        "success": False,
        "deleted_faces": [],
        "error": None,
        "uat_mode": UAT_MODE,
        "skip_aws": SKIP_AWS_CHECKS
    }

    # Skip if AWS checks are disabled (no credentials)
    if SKIP_AWS_CHECKS:
        result["success"] = True
        result["deleted_faces"] = ["AWS_CHECKS_SKIPPED"]
        result["error"] = None
        logger.warning(f"[SKIP AWS] Skipping face deletion for {face_id} - AWS checks disabled")
        return result

    # Skip deletion in UAT mode
    if UAT_MODE:
        result["success"] = True
        result["deleted_faces"] = ["UAT_MODE_SKIPPED"]
        result["error"] = None
        logger.warning(f"[UAT MODE] Skipping face deletion for {face_id} - write operations disabled")
        return result

    try:
        rekognition = get_rekognition_client()
        if rekognition is None:
            result["error"] = "Rekognition client not available"
            return result

        collections_to_check = [collection_id] if collection_id else [REKOGNITION_COLLECTION_1, REKOGNITION_COLLECTION_2]

        for coll_id in collections_to_check:
            try:
                response = rekognition.delete_faces(
                    CollectionId=coll_id,
                    FaceIds=[face_id]
                )

                deleted = response.get('DeletedFaces', [])
                if deleted:
                    result["deleted_faces"].extend(deleted)
                    result["success"] = True
                    logger.info(f"[Rekognition] Deleted face {face_id} from {coll_id}")

            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ResourceNotFoundException':
                    logger.info(f"[Rekognition] Collection {coll_id} not found")
                elif error_code == 'InvalidParameterException':
                    logger.info(f"[Rekognition] Face {face_id} not found in {coll_id}")
                else:
                    logger.error(f"[Rekognition] Error deleting from {coll_id}: {e}")
                continue

        if not result["success"] and not result["deleted_faces"]:
            result["error"] = "Face not found in any collection"

        return result

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[Rekognition] Error deleting face: {e}")
        return result


def index_face_to_main_collection(image_bytes: bytes, matri_id: str, s3_key: str = None) -> dict:
    """
    Index a face to the main Rekognition collection (bm_cbs_face_collection).

    This is called after photo approval to add face to the main searchable collection.

    Args:
        image_bytes: The image data as bytes
        matri_id: The user's matri_id
        s3_key: Optional S3 key for the image

    Returns:
        dict with face_id, success status, and any errors
    """
    result = {
        "success": False,
        "face_id": None,
        "faces_indexed": 0,
        "faces_unindexed": 0,
        "error": None,
        "uat_mode": UAT_MODE,
        "skip_aws": SKIP_AWS_CHECKS
    }

    # Skip if AWS checks are disabled (no credentials)
    if SKIP_AWS_CHECKS:
        result["success"] = True
        result["face_id"] = "AWS_CHECKS_SKIPPED"
        result["error"] = None
        logger.warning(f"[SKIP AWS] Skipping face indexing to main collection for {matri_id} - AWS checks disabled")
        return result

    # Skip indexing in UAT mode
    if UAT_MODE:
        result["success"] = True
        result["face_id"] = "UAT_MODE_SKIPPED"
        result["error"] = None
        logger.warning(f"[UAT MODE] Skipping face indexing to main collection for {matri_id} - write operations disabled")
        return result

    try:
        rekognition = get_rekognition_client()
        if rekognition is None:
            result["error"] = "Rekognition client not available"
            return result

        # Use external_image_id as matri_id (matching Lambda behavior)
        external_image_id = matri_id

        # Index to main collection (REKOGNITION_COLLECTION_1 = bm_cbs_face_collection)
        response = rekognition.index_faces(
            CollectionId=REKOGNITION_COLLECTION_1,
            Image={'Bytes': image_bytes},
            ExternalImageId=external_image_id,
            MaxFaces=1,
            QualityFilter="AUTO",
            DetectionAttributes=['ALL']
        )

        faces_indexed = len(response.get('FaceRecords', []))
        faces_unindexed = len(response.get('UnindexedFaces', []))

        result["faces_indexed"] = faces_indexed
        result["faces_unindexed"] = faces_unindexed

        if faces_indexed > 0:
            result["success"] = True
            result["face_id"] = response['FaceRecords'][0]['Face']['FaceId']
            logger.info(f"[Rekognition] Indexed face to main collection for {matri_id}: {result['face_id']}")
        else:
            reasons = [uf.get('Reasons', []) for uf in response.get('UnindexedFaces', [])]
            result["error"] = f"Face not indexed. Reasons: {reasons}"
            logger.error(f"[Rekognition] Failed to index face to main collection for {matri_id}: {reasons}")

        return result

    except ClientError as e:
        error_code = e.response['Error']['Code']
        result["error"] = f"{error_code}: {e.response['Error']['Message']}"
        logger.error(f"[Rekognition] Error indexing face to main collection: {e}")
        return result
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[Rekognition] Unexpected error indexing face to main collection: {e}")
        return result


@app.post("/api/v1/postvalidation")
async def post_validation_index(
    matri_id: str = Form(...),
    status: str = Form(...),  # 'approved' or 'rejected'
    s3_key: str = Form(None),  # S3 key of the photo
    face_id: str = Form(None),  # Rekognition face_id if known
    photo: Optional[UploadFile] = File(None)  # Photo for indexing if not in S3
):
    """
    Post-validation handler for approved/rejected photos.

    **For Approved Photos:**
    - Indexes face to main collection (bm_cbs_face_collection)
    - Photo can be provided via:
      1. Direct upload (photo parameter)
      2. S3 key (will download from S3)

    **For Rejected Photos:**
    - Deletes face from collections if face_id is provided
    - Cleans up any indexed faces for this matri_id

    This endpoint mimics the PostValidationIndex Lambda function behavior.
    """
    start_time = time.time()
    temp_file_path = None

    try:
        # Validate status
        status_lower = status.lower()
        if status_lower not in ['approved', 'rejected']:
            raise HTTPException(
                status_code=400,
                detail="Status must be 'approved' or 'rejected'"
            )

        result = {
            "success": False,
            "matri_id": matri_id,
            "status": status_lower,
            "action_taken": None,
            "details": {}
        }

        if status_lower == 'rejected':
            # ==================== REJECTED PHOTO ====================
            # Delete face from collection if face_id is provided
            if face_id:
                delete_result = delete_face_from_collection(face_id)
                result["action_taken"] = "delete_face"
                result["details"] = {
                    "face_id": face_id,
                    "deleted": delete_result["success"],
                    "deleted_faces": delete_result.get("deleted_faces", []),
                    "error": delete_result.get("error")
                }
                result["success"] = delete_result["success"]

                if delete_result["success"]:
                    logger.info(f"[PostValidation] Deleted face {face_id} for rejected {matri_id}")
                else:
                    logger.error(f"[PostValidation] Failed to delete face for {matri_id}: {delete_result.get('error')}")
            else:
                # No face_id provided - nothing to delete
                result["action_taken"] = "no_action"
                result["details"]["message"] = "No face_id provided for deletion"
                result["success"] = True

        else:
            # ==================== APPROVED PHOTO ====================
            # Index face to main collection
            image_bytes = None

            # Option 1: Photo provided directly
            if photo:
                temp_file_path = save_upload_file_tmp(photo)
                with open(temp_file_path, 'rb') as f:
                    image_bytes = f.read()

            # Option 2: Download from S3
            elif s3_key:
                s3_client = get_s3_client()
                if s3_client:
                    try:
                        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                        image_bytes = response['Body'].read()
                    except ClientError as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download photo from S3: {e.response['Error']['Message']}"
                        )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="S3 client not available"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="For approved photos, either 'photo' or 's3_key' must be provided"
                )

            # Index to main collection
            index_result = index_face_to_main_collection(
                image_bytes=image_bytes,
                matri_id=matri_id,
                s3_key=s3_key
            )

            result["action_taken"] = "index_face"
            result["details"] = {
                "indexed": index_result["success"],
                "face_id": index_result.get("face_id"),
                "collection": REKOGNITION_COLLECTION_1,
                "faces_indexed": index_result.get("faces_indexed", 0),
                "faces_unindexed": index_result.get("faces_unindexed", 0),
                "error": index_result.get("error")
            }
            result["success"] = index_result["success"]

            if index_result["success"]:
                logger.info(f"[PostValidation] Indexed face to main collection for approved {matri_id}")
            else:
                logger.error(f"[PostValidation] Failed to index face for {matri_id}: {index_result.get('error')}")

        # Cleanup temp file if created
        if temp_file_path:
            cleanup_temp_files(temp_file_path)

        response_time = round(time.time() - start_time, 3)
        result["response_time_seconds"] = response_time

        return result

    except HTTPException:
        if temp_file_path:
            cleanup_temp_files(temp_file_path)
        raise
    except Exception as e:
        if temp_file_path:
            cleanup_temp_files(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/face/{matri_id}")
async def delete_face_for_matri(matri_id: str, face_id: str = None):
    """
    Delete indexed face(s) for a matri_id from Rekognition collections.

    - If face_id is provided, deletes that specific face
    - If no face_id, searches and deletes all faces for this matri_id

    Use this for cleanup or when a user account is deleted.
    """
    # Skip if AWS checks are disabled (no credentials)
    if SKIP_AWS_CHECKS:
        logger.warning(f"[SKIP AWS] Skipping face deletion for {matri_id} - AWS checks disabled")
        return {
            "success": True,
            "matri_id": matri_id,
            "deleted_faces_count": 0,
            "deleted_faces": [],
            "errors": None,
            "skip_aws": True,
            "message": "AWS CHECKS DISABLED - deletion skipped, no credentials configured"
        }

    # Skip in UAT mode
    if UAT_MODE:
        logger.warning(f"[UAT MODE] Skipping face deletion for {matri_id} - write operations disabled")
        return {
            "success": True,
            "matri_id": matri_id,
            "deleted_faces_count": 0,
            "deleted_faces": [],
            "errors": None,
            "uat_mode": True,
            "message": "UAT MODE - deletion skipped, no changes made to Rekognition"
        }

    try:
        rekognition = get_rekognition_client()
        if rekognition is None:
            raise HTTPException(status_code=500, detail="Rekognition client not available")

        deleted_faces = []
        errors = []

        if face_id:
            # Delete specific face
            result = delete_face_from_collection(face_id)
            if result["success"]:
                deleted_faces.extend(result["deleted_faces"])
            elif result.get("error"):
                errors.append(result["error"])
        else:
            # Search and delete all faces for this matri_id
            for collection_id in [REKOGNITION_COLLECTION_1, REKOGNITION_COLLECTION_2]:
                try:
                    # List faces and find ones matching this matri_id
                    paginator = rekognition.get_paginator('list_faces')
                    faces_to_delete = []

                    for page in paginator.paginate(CollectionId=collection_id, MaxResults=100):
                        for face in page.get('Faces', []):
                            external_id = face.get('ExternalImageId', '')
                            if external_id.startswith(matri_id):
                                faces_to_delete.append(face['FaceId'])

                    # Delete found faces
                    if faces_to_delete:
                        response = rekognition.delete_faces(
                            CollectionId=collection_id,
                            FaceIds=faces_to_delete
                        )
                        deleted = response.get('DeletedFaces', [])
                        deleted_faces.extend(deleted)
                        logger.info(f"[Rekognition] Deleted {len(deleted)} faces for {matri_id} from {collection_id}")

                except ClientError as e:
                    error_msg = f"Error in {collection_id}: {e.response['Error']['Message']}"
                    errors.append(error_msg)
                    logger.error(f"[Rekognition] {error_msg}")

        return {
            "success": len(deleted_faces) > 0 or (not face_id and len(errors) == 0),
            "matri_id": matri_id,
            "deleted_faces_count": len(deleted_faces),
            "deleted_faces": deleted_faces,
            "errors": errors if errors else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== INFO ENDPOINTS ====================

# @app.get("/")
# async def root():
#     return {
#         "service": "Photo Validation API - Full GPU Acceleration",
#         "version": "3.0.0",
#         "gpu_status": get_gpu_info(),
#         "features": [
#             "InsightFace GPU - Face detection & matching",
#             "DeepFace GPU - Age & ethnicity (PRIMARY)",
#             "NudeNet GPU - NSFW detection (10x faster)",
#             "4x faster than CPU"
#         ]
#     }

# @app.get("/health")
# async def health():
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow().isoformat(),
#         "gpu": get_gpu_info()
#     }

# @app.get("/api/v3/gpu/info")
# async def gpu_info():
#     return get_gpu_info()

# ==================== DATABASE QUERY ENDPOINTS ====================

@app.get("/api/v1/validations/{validation_id}")
async def get_validation(validation_id: str):
    """Get a specific validation result by validation_id"""
    try:
        result = get_validation_by_id(validation_id)
        if result:
            return {
                "success": True,
                "data": result
            }
        else:
            raise HTTPException(status_code=404, detail="Validation not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/validations/matri/{matri_id}")
async def get_validations_for_matri(matri_id: str, limit: int = 100):
    """Get all validation results for a matri_id"""
    try:
        results = get_validations_by_matri_id(matri_id, limit)
        return {
            "success": True,
            "matri_id": matri_id,
            "total_count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/validations/batch/{batch_id}")
async def get_validations_for_batch(batch_id: str):
    """Get all validation results for a batch_id"""
    try:
        results = get_validations_by_batch_id(batch_id)
        return {
            "success": True,
            "batch_id": batch_id,
            "total_count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/validations/statistics")
async def get_statistics(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get validation statistics"""
    try:
        stats = get_validation_statistics(start_date, end_date)
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    temp_dir = os.path.join(tempfile.gettempdir(), "photo_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    logger.info("="*70)
    logger.info("Photo Validation API - Full GPU v3.0.0")
    logger.info("="*70)
    logger.info(f"InsightFace GPU: {ONNX_GPU_AVAILABLE}")
    logger.info(f"DeepFace GPU: {TF_GPU_AVAILABLE}")
    logger.info(f"NudeNet GPU: {ONNX_GPU_AVAILABLE}")
    logger.info("="*70)

    # Initialize PostgreSQL database
    logger.info("[DB] Initializing PostgreSQL connection...")
    db_initialized = initialize_database()
    if db_initialized:
        logger.info("[DB] PostgreSQL database initialized successfully")
    else:
        logger.error("[DB] WARNING: PostgreSQL initialization failed - validation results will not be stored")
    logger.info("="*70)

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)
    # Close database connections
    close_all_connections()
    logger.info("[DB] Database connections closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hybrid_dev_api:app", host="0.0.0.0", port=8001, reload=True, workers=1)
