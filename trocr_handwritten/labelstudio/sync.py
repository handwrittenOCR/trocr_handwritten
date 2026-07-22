import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import boto3
from dotenv import load_dotenv

from trocr_handwritten.labelstudio import LabelStudio, _env
from trocr_handwritten.utils.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)

CONFIG_XML = Path(__file__).with_name("label_config.xml")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_EXTS = (".txt", ".md")


def s3_client():
    """Boto3 client for Scaleway Object Storage (S3-compatible)."""
    from botocore.config import Config

    region = os.environ.get("SCW_REGION", "fr-par")
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get(
            "SCW_S3_ENDPOINT", f"https://s3.{region}.scw.cloud"
        ),
        aws_access_key_id=_env("SCW_ACCESS_KEY"),
        aws_secret_access_key=_env("SCW_SECRET_KEY"),
        region_name=region,
        config=Config(max_pool_connections=32),
    )


def set_bucket_cors(bucket: str) -> None:
    """
    Allow cross-origin GET/HEAD on the bucket so Label Studio can load images.

    Args:
        bucket: Object Storage bucket to configure.
    """
    s3_client().put_bucket_cors(
        Bucket=bucket,
        CORSConfiguration={
            "CORSRules": [
                {
                    "AllowedOrigins": ["*"],
                    "AllowedMethods": ["GET", "HEAD"],
                    "AllowedHeaders": ["*"],
                    "MaxAgeSeconds": 3000,
                }
            ]
        },
    )
    logger.info(f"CORS configured on bucket {bucket}")


def public_url(bucket: str, key: str) -> str:
    """Public object URL on Scaleway Object Storage."""
    region = os.environ.get("SCW_REGION", "fr-par")
    return f"https://{bucket}.s3.{region}.scw.cloud/{key}"


def existing_text(image: Path) -> str:
    """Return the transcription from a sibling .txt/.md file, or '' if none."""
    for ext in TEXT_EXTS:
        sidecar = image.with_suffix(ext)
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()
    return ""


def collect_tasks(images_dir: Path) -> List[Dict]:
    """
    Build one task per image found under ``images_dir`` (recursive).

    The bucket key is the image path relative to ``images_dir`` (so nested folders
    stay unique). A sibling ``.txt``/``.md`` file, if present, pre-fills the
    transcription for correction; otherwise the box starts empty.

    Args:
        images_dir: Folder of images to transcribe.

    Returns:
        List of task dicts: {"image", "key", "text"}.
    """
    tasks = []
    for img in sorted(images_dir.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        key = img.relative_to(images_dir).as_posix()
        tasks.append({"image": str(img), "key": key, "text": existing_text(img)})
    return tasks


def upload_images(
    tasks: List[Dict], bucket: str, dry_run: bool = False, workers: int = 16
) -> None:
    """
    Upload each image to the bucket (public-read) and set its URL in place.

    Uploads run concurrently in a thread pool. Re-running is safe: existing keys
    are overwritten.

    Args:
        tasks: Task records with an "image" file path and a "key".
        bucket: Destination Object Storage bucket.
        dry_run: If True, only compute URLs without uploading.
        workers: Concurrent upload threads.
    """
    for t in tasks:
        t["url"] = public_url(bucket, t["key"])
    if dry_run:
        return

    s3 = s3_client()

    def _put(t):
        s3.upload_file(t["image"], bucket, t["key"], ExtraArgs={"ACL": "public-read"})

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_put, t): t for t in tasks}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 100 == 0 or done == len(tasks):
                logger.info(f"uploaded {done}/{len(tasks)}")


def to_ls_item(t: Dict) -> Dict:
    """
    Build a Label Studio task with the image URL and any pre-filled transcription.

    The text lives in ``data.transcription`` (bound by the TextArea's
    ``value="$transcription"``) so it is editable immediately.
    """
    return {
        "data": {
            "image": t["url"],
            "transcription": t["text"],
            "filename": t["key"],
        }
    }


def main():
    """CLI: upload a folder of images to Object Storage and import them into Label Studio."""
    parser = argparse.ArgumentParser(
        description="Sync images to Label Studio for transcription."
    )
    parser.add_argument("--images", help="Folder of images to transcribe.")
    parser.add_argument("--bucket", default=os.environ.get("SCW_BUCKET"))
    parser.add_argument(
        "--project",
        help="Label Studio project title (default: the images folder name).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Reuse already-uploaded images; only create the project and import tasks.",
    )
    parser.add_argument(
        "--cors-only",
        action="store_true",
        help="Only set the bucket CORS rule (so Label Studio can load images) and exit.",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Push the current label_config.xml to the project and exit.",
    )
    args = parser.parse_args()

    if not args.bucket:
        raise SystemExit("Provide --bucket or set SCW_BUCKET")

    if args.cors_only:
        set_bucket_cors(args.bucket)
        return

    if not args.images:
        raise SystemExit("Provide --images <folder>")
    images_dir = Path(args.images)
    if not images_dir.is_dir():
        raise SystemExit(f"Not a folder: {images_dir}")
    project = args.project or images_dir.resolve().name

    label_config = CONFIG_XML.read_text()

    if args.update_config:
        ls = LabelStudio(_env("LS_URL"), _env("LS_TOKEN"))
        pid = ls.get_or_create_project(project, label_config)
        ls.update_config(pid, label_config)
        logger.info(f"updated config: project {pid} ({project})")
        return

    tasks = collect_tasks(images_dir)
    if not tasks:
        raise SystemExit(f"No images found under {images_dir}")
    logger.info(f"{len(tasks)} images")

    upload_images(tasks, args.bucket, dry_run=args.dry_run or args.skip_upload)

    if args.dry_run:
        logger.info(f"[dry-run] would import {len(tasks)} tasks into '{project}'")
        return

    ls = LabelStudio(_env("LS_URL"), _env("LS_TOKEN"))
    pid = ls.get_or_create_project(project, label_config)
    ls.import_tasks(pid, [to_ls_item(t) for t in tasks])
    logger.info(f"'{project}' -> project {pid}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
