import argparse
import csv
import json
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


def collect_task_file(path: Path, prefix: str = "pages") -> List[Dict]:
    """
    Read prepared review tasks: a JSON list of ``{"data": {...}, "images": [paths]}``.

    Each distinct local image becomes one upload record keyed ``<prefix>/<file name>``,
    shared by every task that shows it.

    Args:
        path: JSON file of prepared tasks.
        prefix: Bucket folder for the page images.

    Returns:
        Tuple (tasks, uploads): the task list and one {"image", "key"} record per image.
    """
    tasks = json.loads(path.read_text(encoding="utf-8"))
    uploads = {
        img: {"image": img, "key": f"{prefix}/{Path(img).name}"}
        for t in tasks
        for img in t["images"]
    }
    return tasks, list(uploads.values())


def to_review_item(t: Dict, urls: Dict[str, str]) -> Dict:
    """
    Build a Label Studio task from a prepared review task, with the uploaded image URLs.

    ``data.images`` lists every page (for ``valueList``) and ``data.image`` the first one.
    """
    pages = [urls[img] for img in t["images"]]
    return {"data": {**t["data"], "images": pages, "image": pages[0]}}


def crop_pages(manifest: Path) -> Dict[str, str]:
    """Map each Label Studio crop file name (``ls_key`` base name) to its source page."""
    with manifest.open(encoding="utf-8") as f:
        return {
            Path(r["ls_key"]).name: r["page"] for r in csv.DictReader(f) if r["ls_key"]
        }


def add_pages(
    ls: LabelStudio,
    pid: int,
    pages_dir: Path,
    manifest: Path,
    bucket: str,
    dry_run: bool = False,
    workers: int = 16,
    skip_upload: bool = False,
) -> None:
    """
    Add the full source page to every task of a project, as ``data.page``.

    The page comes from the manifest by crop file name, or else from the crop name
    prefix ``<page stem>__``. Pages upload to ``pages/<page>``; annotations are kept.

    Args:
        ls: Label Studio client.
        pid: Project id.
        pages_dir: Folder of full pages (data/raw/initial).
        manifest: pages_manifest.csv with ``ls_key`` and ``page``.
        bucket: Object Storage bucket.
        dry_run: If True, only report the matches.
        workers: Concurrent upload threads.
        skip_upload: If True, reuse pages already in the bucket.
    """
    by_crop = crop_pages(manifest)
    tasks = ls.export_tasks(pid, all_tasks=True)
    pages = {}
    for t in tasks:
        name = Path(t["data"]["filename"]).name
        pages[t["id"]] = by_crop.get(name) or name.split("__")[0] + ".jpg"
    missing = {p for p in pages.values() if not (pages_dir / p).exists()}
    if missing:
        raise SystemExit(
            f"{len(missing)} pages not found in {pages_dir}: {sorted(missing)[:5]}"
        )
    uploads = [
        {"image": str(pages_dir / p), "key": f"pages/{p.replace(' ', '_')}"}
        for p in sorted(set(pages.values()))
    ]
    logger.info(f"{len(tasks)} tasks, {len(uploads)} pages")
    upload_images(uploads, bucket, dry_run=dry_run or skip_upload, workers=workers)
    if dry_run:
        return
    urls = {Path(u["image"]).name: u["url"] for u in uploads}
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                ls.update_task, t["id"], {**t["data"], "page": urls[pages[t["id"]]]}
            )
            for t in tasks
        ]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 100 == 0 or done == len(tasks):
                logger.info(f"updated {done}/{len(tasks)} tasks")
    logger.info(f"project {pid}: page added to {len(tasks)} tasks")


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
    parser.add_argument(
        "--tasks",
        help="JSON file of prepared review tasks ({data, images}) instead of --images.",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_XML),
        help="Labeling config XML (default: label_config.xml).",
    )
    parser.add_argument(
        "--pages-dir",
        help="Add the full source page under each crop of --project-id (e.g. data/raw/initial).",
    )
    parser.add_argument("--project-id", type=int, help="Project id for --pages-dir.")
    parser.add_argument(
        "--manifest",
        default="data/processed/pages_manifest.csv",
        help="Region manifest mapping crops (ls_key) to pages, for --pages-dir.",
    )
    parser.add_argument("--bucket", default=os.environ.get("SCW_BUCKET"))
    parser.add_argument(
        "--project",
        help="Label Studio project title (default: the images folder name).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=16, help="Concurrent upload threads."
    )
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

    label_config = Path(args.config).read_text()

    if args.pages_dir:
        if not args.project_id:
            raise SystemExit("Provide --project-id with --pages-dir")
        ls = LabelStudio(_env("LS_URL"), _env("LS_TOKEN"))
        add_pages(
            ls,
            args.project_id,
            Path(args.pages_dir),
            Path(args.manifest),
            args.bucket,
            dry_run=args.dry_run,
            workers=args.workers,
            skip_upload=args.skip_upload,
        )
        if not args.dry_run:
            ls.update_config(args.project_id, label_config)
            logger.info(f"updated config: project {args.project_id}")
        return

    if args.tasks:
        project = args.project or Path(args.tasks).stem
        tasks, uploads = collect_task_file(Path(args.tasks))
        logger.info(f"{len(tasks)} tasks, {len(uploads)} images")
        upload_images(
            uploads,
            args.bucket,
            dry_run=args.dry_run or args.skip_upload,
            workers=args.workers,
        )
        if args.dry_run:
            logger.info(f"[dry-run] would import {len(tasks)} tasks into '{project}'")
            return
        urls = {u["image"]: u["url"] for u in uploads}
        ls = LabelStudio(_env("LS_URL"), _env("LS_TOKEN"))
        pid = ls.get_or_create_project(project, label_config)
        ls.import_tasks(pid, [to_review_item(t, urls) for t in tasks])
        logger.info(f"'{project}' -> project {pid}: {len(tasks)} tasks")
        return

    if not args.images:
        raise SystemExit("Provide --images <folder> or --tasks <file.json>")
    images_dir = Path(args.images)
    if not images_dir.is_dir():
        raise SystemExit(f"Not a folder: {images_dir}")
    project = args.project or images_dir.resolve().name

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

    upload_images(
        tasks,
        args.bucket,
        dry_run=args.dry_run or args.skip_upload,
        workers=args.workers,
    )

    if args.dry_run:
        logger.info(f"[dry-run] would import {len(tasks)} tasks into '{project}'")
        return

    ls = LabelStudio(_env("LS_URL"), _env("LS_TOKEN"))
    pid = ls.get_or_create_project(project, label_config)
    ls.import_tasks(pid, [to_ls_item(t) for t in tasks])
    logger.info(f"'{project}' -> project {pid}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
