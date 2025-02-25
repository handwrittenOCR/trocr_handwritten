import os
import boto3
from pathlib import Path
from typing import Optional, List
from botocore.exceptions import ClientError
import logging
from dataclasses import dataclass
import shutil
from dotenv import load_dotenv


@dataclass
class S3Config:
    """AWS S3 configuration"""

    bucket_name: str
    region_name: str = "eu-west-1"


class FileManager:
    """Manages file operations between local storage and AWS S3"""

    def __init__(
        self,
        local_path: str | Path,
        s3_config: S3Config,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize FileManager

        Args:
            local_path: Local path for data storage
            s3_config: S3 configuration object
            logger: Optional logger instance
        """
        # Load environment variables from .env file
        load_dotenv()

        # Get AWS credentials from environment variables
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError(
                "AWS credentials not found in environment variables. "
                "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file"
            )

        self.local_path = Path(local_path)
        self.s3_config = s3_config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize S3 client
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=s3_config.region_name,
        )

    def download_from_s3(
        self, s3_prefix: str, local_subdir: Optional[str] = None
    ) -> Path:
        """
        Download files from S3 bucket to local storage

        Args:
            s3_prefix: Prefix (folder) in S3 bucket
            local_subdir: Optional subdirectory in local_path

        Returns:
            Path to downloaded files
        """
        # Prepare local directory
        if local_subdir:
            local_dir = self.local_path / local_subdir
        else:
            local_dir = self.local_path
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            # List objects in S3 with the given prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            objects = paginator.paginate(
                Bucket=self.s3_config.bucket_name, Prefix=s3_prefix
            )

            # Download each object
            for page in objects:
                if "Contents" not in page:
                    self.logger.warning(f"No objects found with prefix: {s3_prefix}")
                    continue

                for obj in page["Contents"]:
                    # Get relative path
                    rel_path = obj["Key"][len(s3_prefix) :].lstrip("/")
                    if not rel_path:  # Skip if it's the directory itself
                        continue

                    # Create local file path
                    local_file = local_dir / rel_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    # Download file
                    self.logger.info(f"Downloading {obj['Key']} to {local_file}")
                    self.s3_client.download_file(
                        self.s3_config.bucket_name, obj["Key"], str(local_file)
                    )

            return local_dir

        except ClientError as e:
            self.logger.error(f"Error downloading from S3: {e}")
            raise

    def upload_to_s3(
        self, local_dir: str | Path, s3_prefix: str, include_patterns: List[str] = None
    ) -> None:
        """
        Upload local directory to S3

        Args:
            local_dir: Local directory to upload
            s3_prefix: Prefix (folder) in S3 bucket
            include_patterns: List of file patterns to include (e.g., ["*.jpg", "*.json"])
        """
        local_dir = Path(local_dir)
        if not local_dir.exists():
            raise FileNotFoundError(f"Directory not found: {local_dir}")

        try:
            # Walk through directory
            for root, _, files in os.walk(local_dir):
                for file in files:
                    # Check if file matches include patterns
                    if include_patterns and not any(
                        file.endswith(pattern.replace("*", ""))
                        for pattern in include_patterns
                    ):
                        continue

                    local_file = Path(root) / file
                    # Calculate S3 key
                    rel_path = local_file.relative_to(local_dir)
                    s3_key = f"{s3_prefix.rstrip('/')}/{rel_path}"

                    # Upload file
                    self.logger.info(
                        f"Uploading {local_file} to s3://{self.s3_config.bucket_name}/{s3_key}"
                    )
                    self.s3_client.upload_file(
                        str(local_file), self.s3_config.bucket_name, s3_key
                    )

        except ClientError as e:
            self.logger.error(f"Error uploading to S3: {e}")
            raise

    def clean_local_directory(self, subdir: Optional[str] = None) -> None:
        """
        Clean local directory

        Args:
            subdir: Optional subdirectory to clean
        """
        try:
            dir_to_clean = self.local_path / subdir if subdir else self.local_path
            if dir_to_clean.exists():
                shutil.rmtree(dir_to_clean)
                self.logger.info(f"Cleaned directory: {dir_to_clean}")
        except Exception as e:
            self.logger.error(f"Error cleaning directory: {e}")
            raise
