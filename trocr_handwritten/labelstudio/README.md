# Transcribe images with Label Studio

Launch a Label Studio server and load any folder of images to transcribe. Each image becomes one task with an editable text box: type the transcription from scratch, or correct a pre-filled one. Export pulls the finished transcriptions back as plain `.txt` files.

Pre-fill is optional: if a `<name>.txt` or `<name>.md` file sits next to `<name>.jpg`, its text seeds that image's box; otherwise the box starts empty.

Prerequisites (local): `uv sync` to install `boto3`, `requests`, and `python-dotenv`.

## 1. Object Storage (images)

Create a Scaleway Object Storage bucket (region `fr-par`), e.g. `transcription`. Generate API keys (Access Key / Secret Key). Use this same bucket name for `SCW_BUCKET` in step 3. Images are stored here so the browser can display them in Label Studio.

## 2. Label Studio server — DEV1-S instance (cheapest, ~€6.34/mo)

Create a DEV1-S Instance (Ubuntu), then on it:

```bash
curl -fsSL https://get.docker.com | sh
git clone <this repo> && cd trocr_handwritten/trocr_handwritten/labelstudio
cat > .env <<EOF
POSTGRES_PASSWORD=$(openssl rand -hex 16)
LABEL_STUDIO_HOST=http://<instance-public-ip>:8080
LABEL_STUDIO_USERNAME=you@example.com
LABEL_STUDIO_PASSWORD=$(openssl rand -hex 12)
EOF
docker compose up -d
```

Open `http://<instance-ip>:8080`, log in, and copy your API token from *Account & Settings → Access Token*. (Open port 8080 in the instance security group.)

## 3. Upload images + create the project (run locally)

```bash
export LS_URL=http://<instance-ip>:8080
export LS_TOKEN=<your-token>
export SCW_BUCKET=transcription
export SCW_ACCESS_KEY=... SCW_SECRET_KEY=... SCW_REGION=fr-par

python -m trocr_handwritten.labelstudio.sync --cors-only              # once per bucket
python -m trocr_handwritten.labelstudio.sync --images ./my_images     # add --dry-run to preview
```

`--cors-only` sets the bucket CORS rule so the browser can load the images; run it once per bucket. The main run uploads every image found under `--images` (recursively, public-read) and creates a Label Studio project named after the folder (override with `--project "My project"`). Open the project and transcribe; tick `reject` for an image that should be dropped.

Useful flags: `--skip-upload` (reuse already-uploaded images), `--update-config` (re-push `label_config.xml` to the project and exit).

## 4. Export transcriptions

Find the project id in the Label Studio UI, then:

```bash
export LS_URL=... LS_TOKEN=...
python -m trocr_handwritten.labelstudio.export --projects 1 --out ./transcriptions
```

Writes one `<name>.txt` per transcribed image into `--out`, plus a `transcriptions.json` list of `{filename, text}`. Rejected images and empty boxes are skipped. Pass several ids to `--projects` to merge multiple projects.

## Notes
- Images are uploaded public-read for durable URLs. For a private bucket, switch to presigned URLs or a Label Studio S3 source storage connection.
- 2 GB RAM (DEV1-S) suffices for 1–2 annotators; bump to DEV1-M (4 GB) for larger batches.
- Cost: DEV1-S ~€6.34/mo + Object Storage (~€0, within the 75 GB free tier).
