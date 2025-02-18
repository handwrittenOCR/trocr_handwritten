import click
from pathlib import Path
from .file_manager import FileManager, S3Config
import logging


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


@click.group()
def cli():
    """CLI tool for managing AWS S3 data transfers"""
    pass


@cli.command()
@click.option("--bucket", required=True, help="S3 bucket name")
@click.option("--s3-prefix", required=True, help="S3 prefix (folder) to download from")
@click.option(
    "--local-dir",
    default="data",
    help="Local directory to store files (default: data)",
)
@click.option(
    "--local-subdir",
    default=None,
    help="Optional subdirectory within local-dir",
)
@click.option("--region", default="eu-west-1", help="AWS region (default: eu-west-1)")
def download(bucket, s3_prefix, local_dir, local_subdir, region):
    """Download files from S3 bucket to local storage"""
    logger = setup_logging()

    s3_config = S3Config(bucket_name=bucket, region_name=region)
    file_manager = FileManager(local_dir, s3_config, logger=logger)

    try:
        local_path = file_manager.download_from_s3(s3_prefix, local_subdir)
        logger.info(f"Successfully downloaded files to {local_path}")
    except Exception as e:
        logger.error(f"Failed to download files: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option("--bucket", required=True, help="S3 bucket name")
@click.option("--s3-prefix", required=True, help="S3 prefix (folder) to upload to")
@click.option(
    "--local-dir",
    required=True,
    help="Local directory to upload",
)
@click.option(
    "--patterns",
    multiple=True,
    default=["*.jpg", "*.json"],
    help="File patterns to include (can be specified multiple times)",
)
@click.option("--region", default="eu-west-1", help="AWS region (default: eu-west-1)")
def upload(bucket, s3_prefix, local_dir, patterns, region):
    """Upload local directory to S3 bucket"""
    logger = setup_logging()

    s3_config = S3Config(bucket_name=bucket, region_name=region)
    file_manager = FileManager(Path(local_dir).parent, s3_config, logger=logger)

    try:
        file_manager.upload_to_s3(local_dir, s3_prefix, list(patterns))
        logger.info(f"Successfully uploaded files to s3://{bucket}/{s3_prefix}")
    except Exception as e:
        logger.error(f"Failed to upload files: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--local-dir",
    required=True,
    help="Local directory to clean",
)
@click.option(
    "--subdir",
    default=None,
    help="Optional subdirectory to clean",
)
def clean(local_dir, subdir):
    """Clean local directory"""
    logger = setup_logging()

    # We need a dummy S3Config since FileManager requires it
    s3_config = S3Config(bucket_name="dummy")
    file_manager = FileManager(local_dir, s3_config, logger=logger)

    try:
        file_manager.clean_local_directory(subdir)
        logger.info("Successfully cleaned directory")
    except Exception as e:
        logger.error(f"Failed to clean directory: {e}")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    cli()
