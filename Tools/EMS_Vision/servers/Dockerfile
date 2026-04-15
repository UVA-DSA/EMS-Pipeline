FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/workspace/EgoEMS
ENV BACKEND=detr-trt
ENV ENGINE_PATH=/workspace/EgoEMS/Tools/inference/checkpoints/ems_finetuned_detr_trt.ts
ENV ACTIVITY_ENGINE_PATH=/workspace/EgoEMS/Tools/inference/checkpoints/mtrsap_30frames_window_resnet_trt.ts
ENV ACTIVITY_FEATURE_ENGINE_PATH=/workspace/EgoEMS/Tools/inference/checkpoints/resnet50_feature_extractor_trt.ts
ENV ACTIVITY_CLASS_MAP_PATH=/workspace/EgoEMS/Tools/inference/conversion/action_class_mapping.json
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV DEVICE=cuda
ENV WARMUP=3
ENV DETR_VERSION=ems
ENV DETECTION_THRESHOLD=0.7
ENV ENGINE_HEIGHT=480
ENV ENGINE_WIDTH=640
ENV ACTIVITY_WINDOW_SIZE=30
ENV ACTIVITY_STRIDE=1
ENV ACTIVITY_MODEL_SEQ_LEN=
ENV ACTIVITY_RESIZE_SHORT_SIDE=256
ENV ACTIVITY_CENTER_CROP_SIZE=224
ENV ACTIVITY_FEATURE_WEIGHTS=imagenet1k_v1

WORKDIR /opt/inference-server

COPY servers/start_inference_server.sh /usr/local/bin/start-inference-server.sh
COPY checkpoints/ems_finetuned_detr_trt.ts /workspace/EgoEMS/Tools/inference/checkpoints/ems_finetuned_detr_trt.ts
COPY checkpoints/mtrsap_30frames_window_resnet_trt.ts /workspace/EgoEMS/Tools/inference/checkpoints/mtrsap_30frames_window_resnet_trt.ts
COPY checkpoints/resnet50_feature_extractor_trt.ts /workspace/EgoEMS/Tools/inference/checkpoints/resnet50_feature_extractor_trt.ts
COPY conversion/action_class_mapping.json /workspace/EgoEMS/Tools/inference/conversion/action_class_mapping.json
COPY servers/container_inference_server.py /workspace/EgoEMS/Tools/inference/servers/container_inference_server.py
COPY servers/container_inference /workspace/EgoEMS/Tools/inference/servers/container_inference

RUN chmod +x /usr/local/bin/start-inference-server.sh \
    && python -m pip install --no-cache-dir aiohttp opencv-python-headless

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/start-inference-server.sh"]
