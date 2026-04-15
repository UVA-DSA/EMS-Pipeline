# Container Inference Server

This README explains how to run and operate the containerized inference server under [Tools/inference/servers](/mnt/f/repos/EgoEMS/Tools/inference/servers).

The server supports two separate HTTP inference paths:

- object detection with DETR TensorRT
- activity recognition with MTRSAP TensorRT plus per-frame ResNet feature extraction

These paths are intentionally separate:

- DETR is fast and frame-based
- MTRSAP is slower and window-based
- the client sends requests to different endpoints
- DETR responses are not delayed by activity buffering or activity inference

Use this document when you need to:

- set up inference on a new machine using a prebuilt image
- run DETR-only or DETR-plus-activity modes
- call inference endpoints and interpret response states
- troubleshoot common runtime issues

Start here:

1. New machine, no image build: [New PC Setup (Use Prebuilt Inference Container)](#new-pc-setup-use-prebuilt-inference-container)
2. Build image locally: [Build From Source (Optional)](#build-from-source-optional)
3. API usage details: [Object Detection](#object-detection) and [Activity Recognition](#activity-recognition)

## Table Of Contents

- [Overview](#overview)
- [Repository Files](#repository-files)
- [Architecture](#architecture)
- [New PC Setup (Use Prebuilt Inference Container)](#new-pc-setup-use-prebuilt-inference-container)
  - [New PC Setup: Host Requirements](#host-requirements)
  - [New PC Setup: Verify Docker And GPU Runtime](#verify-docker-and-gpu-runtime)
  - [New PC Setup: Pull The Prebuilt Image](#pull-the-prebuilt-image)
  - [New PC Setup: Start The Inference Server](#start-the-inference-server)
  - [New PC Setup: Validate The Server](#validate-the-server)
  - [New PC Setup: Basic Operations](#basic-operations)
- [Build From Source (Optional)](#build-from-source-optional)
  - [Build From Source: Convert Checkpoints To TensorRT Engines](#convert-checkpoints-to-tensorrt-engines)
  - [Build From Source: Build Context](#build-context)
  - [Build From Source: Run The Rebuilt Image](#run-the-rebuilt-image)
- [Run Modes](#run-modes)
- [Health Check](#health-check)
- [Object Detection](#object-detection)
- [Activity Recognition](#activity-recognition)
- [Client Pattern](#client-pattern)
- [Configuration](#configuration)
- [Build, Push, Pull](#build-push-pull)
- [Logs And Debugging](#logs-and-debugging)
- [Quick Validation With curl](#quick-validation-with-curl)
- [Quick Validation: Test DETR](#test-detr)
- [Quick Validation: Test Activity For One Stream](#test-activity-for-one-stream)
- [Test Suggestions](#test-suggestions)
- [Troubleshooting](#troubleshooting)
- [Troubleshooting: The container exits immediately](#the-container-exits-immediately)

## Overview

At a high level:

- the Docker container runs one `aiohttp` server
- DETR loads once at startup and serves frame-wise inference
- activity recognition maintains a separate per-stream feature buffer
- each activity request adds one frame, extracts one ResNet feature, and appends it to the stream buffer
- once the activity buffer reaches `T` feature steps, the server runs the MTRSAP TensorRT model on the latest window

Current implementation details:

- DETR is TensorRT inside the container
- MTRSAP is TensorRT inside the container
- activity feature extraction can run through the baked ResNet50 TensorRT engine
- MTRSAP buffering is keyed by `stream_id`
- activity inference returns its own response object and status

## Repository Files

Main server files:

- [Dockerfile](/mnt/f/repos/EgoEMS/Tools/inference/servers/Dockerfile)
- [start_inference_server.sh](/mnt/f/repos/EgoEMS/Tools/inference/servers/start_inference_server.sh)
- [container_inference_server.py](/mnt/f/repos/EgoEMS/Tools/inference/servers/container_inference_server.py)
- [app.py](/mnt/f/repos/EgoEMS/Tools/inference/servers/container_inference/app.py)
- [detr_trt.py](/mnt/f/repos/EgoEMS/Tools/inference/servers/container_inference/backends/detr_trt.py)
- [mtrsap_trt.py](/mnt/f/repos/EgoEMS/Tools/inference/servers/container_inference/backends/mtrsap_trt.py)
- [activity_state.py](/mnt/f/repos/EgoEMS/Tools/inference/servers/container_inference/activity_state.py)
- [types.py](/mnt/f/repos/EgoEMS/Tools/inference/servers/container_inference/types.py)

Useful supporting files:

- [tensorRT_test.py](/mnt/f/repos/EgoEMS/Tools/inference/tests/tensorRT_test.py)
- [test_video_client.py](/mnt/f/repos/EgoEMS/Tools/inference/tests/test_video_client.py)
- [action_class_mapping.json](/mnt/f/repos/EgoEMS/Tools/inference/conversion/action_class_mapping.json)

## Architecture

What runs where:

- host machine
  - your client program
  - optional test scripts
  - JPEG encoding or frame capture
  - HTTP requests to the container
- Docker container
  - DETR TensorRT engine
  - MTRSAP TensorRT engine, if enabled
  - ResNet50 activity feature extractor, if activity is enabled
  - per-stream activity feature buffers
  - `/health`, DETR, and activity endpoints

The current image is best understood as:

- DETR-first by default
- activity-capable when you provide an activity engine path

## New PC Setup (Use Prebuilt Inference Container)

Use this path when you want to run inference on a new machine without rebuilding images.

This section covers:

- host requirements
- Docker and GPU runtime checks
- pulling the prebuilt image
- running and validating the inference server

This section does not cover model conversion or local image builds.

### Host Requirements

1. NVIDIA GPU driver is installed and `nvidia-smi` works on the host.
2. Docker Engine or Docker Desktop is installed and running.
3. NVIDIA Container Toolkit is installed and configured for Docker GPU access.
4. The machine can pull `keshara2032/egoems-inference-server:latest` from Docker Hub.

### Verify Docker And GPU Runtime

Run:

```bash
docker --version
docker run --rm hello-world
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

If the third command fails, fix Docker + NVIDIA runtime integration before continuing.

### Pull The Prebuilt Image

```bash
docker login
docker pull keshara2032/egoems-inference-server:latest
```

If your Docker Hub access is public, `docker login` can be skipped.

### Start The Inference Server

DETR only:

```bash
docker run --gpus all --rm -d \
  --name egoems-inference-server \
  -p 8000:8000 \
  -e ACTIVITY_ENGINE_PATH= \
  -e ACTIVITY_FEATURE_ENGINE_PATH= \
  keshara2032/egoems-inference-server:latest
```

DETR plus activity (uses baked default engines in the image):

```bash
docker run --gpus all --rm -d \
  --name egoems-inference-server \
  -p 8000:8000 \
  keshara2032/egoems-inference-server:latest
```

### Validate The Server

Health check:

```bash
curl http://localhost:8000/health
```

DETR test:

```bash
curl -X POST http://localhost:8000/infer/detr \
  -H "Content-Type: application/octet-stream" \
  --data-binary @/path/to/test_frame.jpg
```

Activity test:

```bash
curl -X POST http://localhost:8000/infer/activity/cam_01 \
  -H "Content-Type: application/octet-stream" \
  --data-binary @/path/to/test_frame.jpg
```

### Basic Operations

Follow logs:

```bash
docker logs -f egoems-inference-server
```

Stop container:

```bash
docker stop egoems-inference-server
```

## Build From Source (Optional)

Use this path when you want a complete source-based workflow:

- convert checkpoints to TensorRT engines on the target GPU class
- bake those engines into the Docker image
- run inference from the rebuilt image

### Prerequisites

Before conversion or image build:

1. Docker is installed.
2. NVIDIA Container Toolkit is working.
3. `docker run --gpus all ...` works on the machine.
4. You have the source checkpoints:
   - `/path/to/EgoEMS/Tools/inference/checkpoints/ems_finetuned_detr_checkpoint.pth`
   - `/path/to/EgoEMS/Tools/inference/checkpoints/mtrsap_30frames_window_resnet.pt` (if activity is needed)

If you want activity recognition enabled, you also need:

1. an MTRSAP checkpoint (`.pt`) to convert
2. the correct model sequence length if the engine expects a fixed `T`
3. enough GPU memory for:
   - DETR TensorRT
   - MTRSAP TensorRT
   - ResNet50 feature extractor

Important:

- TensorRT engines are GPU-compatibility sensitive.
- Build/convert on the same GPU family you will deploy on.
- Example: engines compiled on RTX 3060 (Ampere) may fail on RTX 4080 (Ada).

### Convert Checkpoints To TensorRT Engines

Run conversion on a CUDA/TensorRT-capable runtime on the target machine.

From repo root:

```bash
cd /path/to/EgoEMS
docker run --rm --gpus all -it \
  -v "$PWD":/workspace/EgoEMS \
  -w /workspace/EgoEMS/Tools/inference \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash
```

Inside that container:

```bash
python conversion/DETR_model_to_tensorRT.py \
  --checkpoint checkpoints/ems_finetuned_detr_checkpoint.pth \
  --output checkpoints/ems_finetuned_detr_trt.ts \
  --detr-version ems \
  --height 480 --width 640 \
  --ir auto \
  --fp16
```

If activity is enabled:

```bash
python conversion/MTRSAP_model_to_tensorRT.py \
  --checkpoint checkpoints/mtrsap_30frames_window_resnet.pt \
  --output checkpoints/mtrsap_30frames_window_resnet_trt.ts \
  --seq-len 30 \
  --nhead 4 \
  --fp16

python conversion/ResNet50_to_tensorRT.py \
  --output checkpoints/resnet50_feature_extractor_trt.ts \
  --weights imagenet1k_v1 \
  --height 224 --width 224 \
  --fp16
```

Verify generated artifacts:

```bash
ls -lh checkpoints/*trt.ts
```

### Build Context

The Dockerfile lives in `Tools/inference/servers`, but the build context must be `Tools/inference`.

Use:

```bash
docker build \
  -f /path/to/EgoEMS/Tools/inference/servers/Dockerfile \
  -t keshara2032/egoems-inference-server:latest \
  /path/to/EgoEMS/Tools/inference
```

Do not use `Tools/inference/servers` as the build context. It is too small.

### Run The Rebuilt Image

After build, run either:

- [Mode 1: DETR Only](#mode-1-detr-only)
- [Mode 2: DETR Plus Activity](#mode-2-detr-plus-activity)

### What The Current Image Bakes In

The current image bakes in:

- server code under `Tools/inference/servers/...`
- the default DETR TensorRT engine
- the default MTRSAP TensorRT engine
- the default ResNet50 TensorRT feature extractor

That means:

- DETR works out of the box after build
- activity also works out of the box after build using the baked TRT engines
- if you want a different activity model, override the activity engine paths at runtime

## Run Modes

There are two practical ways to run the server.

### Mode 1: DETR Only

This keeps the container lightweight at runtime if you only want DETR.

Run:

```bash
docker run --gpus all --rm -d \
  --name egoems-inference-server \
  -p 8000:8000 \
  -e ACTIVITY_ENGINE_PATH= \
  -e ACTIVITY_FEATURE_ENGINE_PATH= \
  keshara2032/egoems-inference-server:latest
```

This starts:

- DETR backend enabled
- activity backend disabled

### Mode 2: DETR Plus Activity

This is now the default production path because the image bakes all three TRT engines.

```bash
docker run --gpus all --rm -d \
  --name egoems-inference-server \
  -p 8000:8000 \
  keshara2032/egoems-inference-server:latest
```

The baked defaults are:

- `ACTIVITY_ENGINE_PATH=/workspace/EgoEMS/Tools/inference/checkpoints/mtrsap_30frames_window_resnet_trt.ts`
- `ACTIVITY_FEATURE_ENGINE_PATH=/workspace/EgoEMS/Tools/inference/checkpoints/resnet50_feature_extractor_trt.ts`
- `ACTIVITY_WINDOW_SIZE`
- `ACTIVITY_MODEL_SEQ_LEN`
- `ACTIVITY_STRIDE`

If you want to swap in another activity model at runtime, override:

- `ACTIVITY_ENGINE_PATH`
- `ACTIVITY_FEATURE_ENGINE_PATH`

If your activity engine was compiled for a fixed `T`, `ACTIVITY_MODEL_SEQ_LEN` should usually be that exact value.

## Health Check

Check server health:

```bash
curl http://localhost:8000/health
```

When DETR only is enabled, the response looks like:

```json
{
  "status": "ok",
  "backends": {
    "detr": {
      "model_name": "detr_tensorrt",
      "engine_path": "/workspace/EgoEMS/Tools/inference/checkpoints/ems_finetuned_detr_trt.ts",
      "detr_version": "ems",
      "threshold": 0.7,
      "engine_height": 480,
      "engine_width": 640,
      "device": "cuda"
    },
    "activity": {
      "enabled": false
    }
  }
}
```

When activity is enabled, the health response also includes:

- activity engine path
- activity window size
- activity stride
- feature extractor metadata
- currently active activity streams

## Object Detection

### Endpoint

DETR endpoints:

- `POST /infer`
- `POST /infer/detr`

Both routes do the same thing.

### Input

Accepted request formats:

- raw image bytes in the request body
- `multipart/form-data`

Supported optional metadata:

- `frame_id`
- `timestamp`

For raw requests:

- `x-frame-id`
- `x-timestamp`

For multipart requests:

- field `image`
- optional field `frame_id`
- optional field `timestamp`

### Example Request

```bash
curl -X POST http://localhost:8000/infer/detr \
  -H "Content-Type: application/octet-stream" \
  -H "x-frame-id: 1" \
  -H "x-timestamp: 1710200000.123456" \
  --data-binary @/path/to/test_frame.jpg
```

### Example Response

```json
{
  "model_name": "detr_tensorrt",
  "frame_id": "1",
  "timestamp": "1710200000.123456",
  "image_width": 1280,
  "image_height": 720,
  "inference_ms": 12.3,
  "preprocess_ms": 2.1,
  "postprocess_ms": 0.8,
  "detections": [
    {
      "label": "bvm",
      "class_id": 2,
      "score": 0.94,
      "box_xyxy": [101.2, 119.8, 240.6, 300.1]
    }
  ]
}
```

### What Happens Internally

For each DETR request:

1. the server decodes the image
2. the frame is resized to the DETR compile-time input size
3. the frame is normalized
4. the TensorRT module runs once
5. logits and boxes are postprocessed
6. detections are returned immediately

DETR does not interact with the activity buffer.

## Activity Recognition

### Endpoint

Activity endpoint:

- `POST /infer/activity/{stream_id}`

Reset endpoint:

- `POST /infer/activity/{stream_id}/reset`

The client should send frames belonging to the same temporal sequence using the same `stream_id`.

### Input

The activity endpoint accepts the same image payload formats as DETR:

- raw image bytes
- `multipart/form-data`

It also accepts the same optional metadata:

- `frame_id`
- `timestamp`

### Example Request

```bash
curl -X POST http://localhost:8000/infer/activity/cam_01 \
  -H "Content-Type: application/octet-stream" \
  -H "x-frame-id: 42" \
  -H "x-timestamp: 1710200001.250000" \
  --data-binary @/path/to/test_frame.jpg
```

### Buffering Behavior

This endpoint is stateful by `stream_id`.

For each request:

1. the server decodes the image
2. one ResNet50 feature is extracted for that single frame
3. the feature is appended to the `stream_id` buffer
4. if the buffer has fewer than `T` features, the response is `buffering`
5. once the buffer reaches `T`, the latest `T` features are stacked into a window
6. the MTRSAP TensorRT model runs on that feature window
7. the activity prediction is returned

The feature buffer is sliding, not one-shot:

- new features are appended continuously
- only the latest `T` are retained
- activity inference can run repeatedly on successive windows

### Stride Behavior

`ACTIVITY_STRIDE` controls how often the activity backend runs after the buffer is full.

Examples:

- `ACTIVITY_STRIDE=1`
  - infer on every new frame after the buffer is full
- `ACTIVITY_STRIDE=4`
  - infer once every 4 newly appended features

### Response States

The activity endpoint can return three statuses:

- `buffering`
- `stride_wait`
- `ready`

`buffering` means:

- the stream has not accumulated `T` features yet

`stride_wait` means:

- the stream buffer is full
- but the configured stride has not elapsed since the last activity inference

`ready` means:

- the backend ran MTRSAP and produced a prediction

### Example Buffering Response

```json
{
  "model_name": "mtrsap_tensorrt",
  "stream_id": "cam_01",
  "frame_id": "42",
  "timestamp": "1710200001.250000",
  "status": "buffering",
  "buffer_size": 19,
  "window_size": 30,
  "stride": 1,
  "feature_dim": 2048,
  "frames_seen": 19,
  "feature_extraction_ms": 6.7,
  "inference_ms": null,
  "activity": null
}
```

### Example Ready Response

```json
{
  "model_name": "mtrsap_tensorrt",
  "stream_id": "cam_01",
  "frame_id": "53",
  "timestamp": "1710200001.683333",
  "status": "ready",
  "buffer_size": 30,
  "window_size": 30,
  "stride": 1,
  "feature_dim": 2048,
  "frames_seen": 30,
  "feature_extraction_ms": 6.9,
  "inference_ms": 41.5,
  "activity": {
    "label": "check_pulse",
    "class_id": 2,
    "score": 0.91,
    "window_size": 30,
    "feature_dim": 2048
  }
}
```

### What Happens Internally

Important current behavior:

- the activity model itself is TensorRT
- the feature extractor can run as TensorRT
- features are buffered on CPU after extraction
- the full activity window is moved back to the model device for each MTRSAP inference

The intended production path is:

- preprocess one frame in Python
- run the ResNet50 TensorRT feature extractor on that frame
- append the `2048`-dim feature to the stream buffer
- run the MTRSAP TensorRT engine on the latest `T` buffered features

### Stream Reset

If you want to clear activity state for one stream:

```bash
curl -X POST http://localhost:8000/infer/activity/cam_01/reset
```

Example response:

```json
{
  "stream_id": "cam_01",
  "cleared": true
}
```

## Client Pattern

The intended client behavior is:

1. capture one frame
2. send the frame to `POST /infer/detr`
3. send the same frame to `POST /infer/activity/{stream_id}`
4. consume the responses independently

This gives:

- fast DETR responses immediately
- slower, buffered activity predictions independently

The server does not fuse these responses for you.

## Configuration

### Shared Environment Variables

- `APP_HOME`
- `BACKEND`
- `ENGINE_PATH`
- `SERVER_HOST`
- `SERVER_PORT`
- `DEVICE`
- `WARMUP`
- `DETR_VERSION`
- `DETECTION_THRESHOLD`
- `ENGINE_HEIGHT`
- `ENGINE_WIDTH`

### Activity Environment Variables

- `ACTIVITY_ENGINE_PATH`
- `ACTIVITY_FEATURE_ENGINE_PATH`
- `ACTIVITY_CLASS_MAP_PATH`
- `ACTIVITY_WINDOW_SIZE`
- `ACTIVITY_STRIDE`
- `ACTIVITY_MODEL_SEQ_LEN`
- `ACTIVITY_RESIZE_SHORT_SIDE`
- `ACTIVITY_CENTER_CROP_SIZE`
- `ACTIVITY_FEATURE_WEIGHTS`

### Example DETR-Only Override

```bash
docker run --gpus all --rm -d \
  --name egoems-inference-server \
  -p 8000:8000 \
  -e DETECTION_THRESHOLD=0.6 \
  -e WARMUP=5 \
  keshara2032/egoems-inference-server:latest
```

### Example DETR Plus Activity Override

```bash
docker run --gpus all --rm -d \
  --name egoems-inference-server \
  -p 8000:8000 \
  -e ACTIVITY_WINDOW_SIZE=30 \
  -e ACTIVITY_MODEL_SEQ_LEN=30 \
  -e ACTIVITY_STRIDE=1 \
  -e ACTIVITY_RESIZE_SHORT_SIDE=256 \
  -e ACTIVITY_CENTER_CROP_SIZE=224 \
  keshara2032/egoems-inference-server:latest
```

## Build, Push, Pull

Build:

```bash
docker build \
  -f /mnt/f/repos/EgoEMS/Tools/inference/servers/Dockerfile \
  -t keshara2032/egoems-inference-server:latest \
  /mnt/f/repos/EgoEMS/Tools/inference
```

Push:

```bash
docker login
docker push keshara2032/egoems-inference-server:latest
```

Optional stable tag:

```bash
docker tag \
  keshara2032/egoems-inference-server:latest \
  keshara2032/egoems-inference-server:v1

docker push keshara2032/egoems-inference-server:v1
```

Pull on another machine:

```bash
docker login
docker pull keshara2032/egoems-inference-server:latest
```

## Logs And Debugging

Follow logs:

```bash
docker logs -f egoems-inference-server
```

Open a shell instead of starting the server:

```bash
docker run --gpus all --rm -it \
  keshara2032/egoems-inference-server:latest \
  bash
```

Because the entrypoint script respects an explicit command, `bash` opens a shell instead of launching the server.

## Quick Validation With curl

After the container is running, you can test both endpoints directly with `curl`.

Assume you have a JPEG frame at:

```text
/path/to/test_frame.jpg
```

### Check Health

```bash
curl http://localhost:8000/health
```

### Test DETR

```bash
curl -X POST http://localhost:8000/infer/detr \
  -H "Content-Type: application/octet-stream" \
  -H "x-frame-id: 1" \
  -H "x-timestamp: 1710200000.123456" \
  --data-binary @/path/to/test_frame.jpg
```

This should return a JSON object with:

- image size
- timing fields
- a `detections` list

### Test Activity For One Stream

Send the same frame repeatedly to build up the activity feature buffer for one stream:

```bash
for i in $(seq 1 35); do
  curl -s -X POST http://localhost:8000/infer/activity/cam_01 \
    -H "Content-Type: application/octet-stream" \
    -H "x-frame-id: ${i}" \
    -H "x-timestamp: 1710200000.${i}" \
    --data-binary @/path/to/test_frame.jpg
  echo
done
```

What you should see:

- early responses with `"status": "buffering"`
- later responses with `"status": "ready"`
- once ready, the response includes the predicted activity class and confidence

### Reset The Activity Buffer For One Stream

```bash
curl -X POST http://localhost:8000/infer/activity/cam_01/reset
```

### Test Multipart Form Instead Of Raw Bytes

DETR:

```bash
curl -X POST http://localhost:8000/infer/detr \
  -F image=@/path/to/test_frame.jpg \
  -F frame_id=1 \
  -F timestamp=1710200000.123456
```

Activity:

```bash
curl -X POST http://localhost:8000/infer/activity/cam_01 \
  -F image=@/path/to/test_frame.jpg \
  -F frame_id=1 \
  -F timestamp=1710200000.123456
```

## Test Suggestions

For DETR:

- run the container in DETR-only mode
- send a single image with `curl`
- confirm `/infer/detr` returns detections or an empty detection list

For activity:

- start the container with `ACTIVITY_ENGINE_PATH`
- choose one `stream_id`
- send repeated frames to `/infer/activity/{stream_id}`
- verify the first responses are `buffering`
- verify the response switches to `ready` once `buffer_size == window_size`

## Troubleshooting

### The container exits immediately

Check logs:

```bash
docker logs egoems-inference-server
```

Common causes:

- the DETR engine path does not exist
- `APP_HOME` was overridden incorrectly
- `ACTIVITY_ENGINE_PATH` was set to a missing file
- CUDA or TensorRT runtime initialization failed

### DETR works but activity is disabled

Check:

- `ACTIVITY_ENGINE_PATH` is set
- the activity engine file exists inside the container
- the path points to the correct `.ts` or `.ep` file

### Activity labels show up as `class_<id>`

This means the class map file was not found or could not be parsed.

Set:

- `ACTIVITY_CLASS_MAP_PATH`

to a valid activity class mapping file if you want human-readable labels.

The baked default is:

- `/workspace/EgoEMS/Tools/inference/conversion/action_class_mapping.json`

### Activity returns errors after the buffer fills

Most likely causes:

- `ACTIVITY_MODEL_SEQ_LEN` does not match the engine compile-time length
- the TRT engine expects a different input feature dimension
- the engine was compiled for a different number of timesteps than your configured window

Use [tensorRT_test.py](/mnt/f/repos/EgoEMS/Tools/inference/tests/tensorRT_test.py) to confirm the expected model sequence length.

### Runtime test from this repo fails because of missing local packages

The container image is the intended runtime. Local shell environments may not have:

- `cv2`
- `torch_tensorrt`
- matching CUDA/TensorRT shared libraries

When in doubt, validate inside the container first.
