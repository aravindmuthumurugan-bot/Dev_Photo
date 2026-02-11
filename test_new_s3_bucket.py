import boto3
import os
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_REGION = os.environ.get("S3_REGION", "ap-south-1")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "bm-photo-validation")

print("=" * 60)
print("S3 BUCKET ACCESS TEST")
print("=" * 60)
print(f"Bucket: {S3_BUCKET_NAME}")
print(f"Region: {S3_REGION}")
print(f"Access Key: {AWS_ACCESS_KEY_ID[:8]}...{AWS_ACCESS_KEY_ID[-4:]}" if AWS_ACCESS_KEY_ID else "Access Key: NOT SET")
print()

try:
    s3 = boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    # 1. Check bucket access
    print("[1] Checking bucket access...")
    s3.head_bucket(Bucket=S3_BUCKET_NAME)
    print("    SUCCESS - Bucket is accessible\n")

    # 2. Get bucket location
    print("[2] Bucket location...")
    location = s3.get_bucket_location(Bucket=S3_BUCKET_NAME)
    print(f"    Region: {location.get('LocationConstraint', 'us-east-1')}\n")

    # 3. List objects and build folder structure
    print("[3] Scanning bucket structure...")
    paginator = s3.get_paginator("list_objects_v2")
    folder_tree = defaultdict(list)
    total_files = 0
    total_size = 0
    extensions = defaultdict(int)
    sample_files = []

    for page in paginator.paginate(Bucket=S3_BUCKET_NAME):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            size = obj["Size"]
            total_files += 1
            total_size += size

            # Track file extensions
            ext = os.path.splitext(key)[1].lower() if "." in key else "(no ext)"
            extensions[ext] += 1

            # Build folder tree (top 2 levels)
            parts = key.split("/")
            if len(parts) >= 2:
                folder_tree[parts[0]].append(key)
            else:
                folder_tree["(root)"].append(key)

            # Collect sample files (first 10)
            if len(sample_files) < 10:
                sample_files.append((key, size))

    print(f"    Total files: {total_files}")
    print(f"    Total size: {total_size / (1024*1024):.2f} MB\n")

    # 4. Show folder structure
    print("[4] Bucket Schema / Folder Structure:")
    print("-" * 50)
    for folder in sorted(folder_tree.keys()):
        file_count = len(folder_tree[folder])
        # Show subfolders
        subfolders = set()
        for f in folder_tree[folder]:
            parts = f.split("/")
            if len(parts) >= 3:
                subfolders.add(parts[1])
        if subfolders:
            print(f"  {folder}/ ({file_count} files)")
            for sf in sorted(list(subfolders)[:10]):
                sub_count = sum(1 for f in folder_tree[folder] if f.startswith(f"{folder}/{sf}/"))
                print(f"    ├── {sf}/ ({sub_count} files)")
            if len(subfolders) > 10:
                print(f"    └── ... and {len(subfolders) - 10} more subfolders")
        else:
            print(f"  {folder}/ ({file_count} files)")
    print()

    # 5. File type breakdown
    print("[5] File Types:")
    print("-" * 50)
    for ext, count in sorted(extensions.items(), key=lambda x: -x[1]):
        print(f"  {ext:15s} : {count} files")
    print()

    # 6. Sample files
    print("[6] Sample Files (first 10):")
    print("-" * 50)
    for key, size in sample_files:
        print(f"  {key} ({size / 1024:.1f} KB)")
    print()

    print("=" * 60)
    print("ALL TESTS PASSED - S3 bucket is accessible")
    print("=" * 60)

except s3.exceptions.NoSuchBucket:
    print(f"  FAILED - Bucket '{S3_BUCKET_NAME}' does not exist")
except Exception as e:
    error_type = type(e).__name__
    print(f"  FAILED - {error_type}: {e}")
