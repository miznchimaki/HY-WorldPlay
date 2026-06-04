#!/usr/bin/bash


source ${HOME}/.bashrc
cd ${HOME}/projects/HY-WorldPlay/
export PYTHONPATH=$(cd "$(dirname "$0")" && pwd):$PYTHONPATH
echo "Environment variable PYTHONPATH: ${PYTHONPATH}"


export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"


# Instruction
PROMPT='In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to Grab the hamburg with printed wrapper and the Palm-sized red fries container, use both arms to set on the flat brown tray with raised edges.'
# Instruction_1
# PROMPT='In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to Use the left arm to pick up the hamburger with the printed wrapper and place it at the far left corner of the table, then use the right arm to move the palm-sized red fries container to the opposite end of the table.'
# Instruction_2
# PROMPT='In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to With both arms, lift the hamburger and fries container in a slow, sweeping motion, and carefully deposit them inside the blue bin.'


IMAGE_DIR=${HOME}/datasets/WorldArena_Robotwin2.0/val_dataset/first_frame/fixed_scene_task
IMAGE_NAME=episode139.png
IMAGE_BASE_NAME="${IMAGE_NAME%.*}"
IMAGE_PATH=${IMAGE_DIR}/${IMAGE_NAME} # Now we only provide the i2v model, so the path cannot be None
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p # Now we only provide the 480p model
MODEL_PATH=${HOME}/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_PATH=${HOME}/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_model/diffusion_pytorch_model.safetensors
AR_RL_ACTION_MODEL_PATH=${HOME}/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_rl_model/diffusion_pytorch_model.safetensors
BI_ACTION_MODEL_PATH=${HOME}/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/bidirectional_model/diffusion_pytorch_model.safetensors
AR_DISTILL_ACTION_MODEL_PATH=${HOME}/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_distilled_action_model/diffusion_pytorch_model.safetensors
# string pose
# POSE='w-31'  # Camera trajectory: pose string (e.g., 'w-31' means generating [1 + 31] latents) or JSON file path

# json file pose
POSE='./assets/pose/test_forward_32_latents.json'
NUM_FRAMES=129
WIDTH=832
HEIGHT=480
# OUTPUT_PATH=./worldarena_text_driven_evaluation_outputs/${IMAGE_BASE_NAME}/"${POSE//, /_}"/instructions
OUTPUT_PATH=./worldarena_text_driven_evaluation_outputs/${IMAGE_BASE_NAME}/json_file_pose/instructions


# Configuration for faster inference
# The maximum number recommended is 8.
N_INFERENCE_GPU=2 # Parallel inference GPU count.


# Configuration for better quality
REWRITE=false   # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution. When the NUM_FRAMES == 125, you can set it to true


# inference with bidirectional model
# qwen3vl-python -m torch.distributed.run --nproc_per_node=${N_INFERENCE_GPU} \
#   --no_python qwen3vl-python hyvideo/generate.py  \
#   --prompt "${PROMPT}" \
#   --image_path ${IMAGE_PATH} \
#   --resolution ${RESOLUTION} \
#   --aspect_ratio ${ASPECT_RATIO} \
#   --video_length ${NUM_FRAMES} \
#   --seed ${SEED} \
#   --rewrite ${REWRITE} \
#   --sr ${ENABLE_SR} --save_pre_sr_video \
#   --pose "${POSE}" \
#   --output_path ${OUTPUT_PATH} \
#   --model_path ${MODEL_PATH} \
#   --action_ckpt ${BI_ACTION_MODEL_PATH} \
#   --few_step false \
#   --model_type 'bi'


# inference with autoregressive model
# qwen3vl-python -m torch.distributed.run --nproc_per_node=${N_INFERENCE_GPU} \
#   --no_python qwen3vl-python hyvideo/generate.py  \
#   --prompt "${PROMPT}" \
#   --image_path ${IMAGE_PATH} \
#   --resolution ${RESOLUTION} \
#   --aspect_ratio ${ASPECT_RATIO} \
#   --video_length ${NUM_FRAMES} \
#   --seed ${SEED} \
#   --rewrite ${REWRITE} \
#   --sr ${ENABLE_SR} --save_pre_sr_video \
#   --pose "${POSE}" \
#   --output_path ${OUTPUT_PATH} \
#   --model_path ${MODEL_PATH} \
#   --action_ckpt ${AR_ACTION_MODEL_PATH} \
#   --few_step false \
#   --width ${WIDTH} \
#   --height ${HEIGHT} \
#   --model_type 'ar'


# inference with autoregressive + RL model
# qwen3vl-python -m torch.distributed.run --nproc_per_node=${N_INFERENCE_GPU} \
#   --no_python qwen3vl-python hyvideo/generate.py  \
#   --prompt "${PROMPT}" \
#   --image_path ${IMAGE_PATH} \
#   --resolution ${RESOLUTION} \
#   --aspect_ratio ${ASPECT_RATIO} \
#   --video_length ${NUM_FRAMES} \
#   --seed ${SEED} \
#   --rewrite ${REWRITE} \
#   --sr ${ENABLE_SR} --save_pre_sr_video \
#   --pose "${POSE}" \
#   --output_path ${OUTPUT_PATH} \
#   --model_path ${MODEL_PATH} \
#   --action_ckpt ${AR_RL_ACTION_MODEL_PATH} \
#   --few_step false \
#   --width ${WIDTH} \
#   --height ${HEIGHT} \
#   --model_type 'ar'


# inference with autoregressive distilled model
qwen3vl-python -m torch.distributed.run --nproc_per_node=${N_INFERENCE_GPU} \
  --no_python qwen3vl-python hyvideo/generate.py \
  --prompt "${PROMPT}" \
  --image_path ${IMAGE_PATH} \
  --resolution ${RESOLUTION} \
  --aspect_ratio ${ASPECT_RATIO} \
  --video_length ${NUM_FRAMES} \
  --seed ${SEED} \
  --rewrite ${REWRITE} \
  --sr ${ENABLE_SR} \
  --save_pre_sr_video \
  --pose "${POSE}" \
  --output_path ${OUTPUT_PATH} \
  --model_path ${MODEL_PATH} \
  --action_ckpt ${AR_DISTILL_ACTION_MODEL_PATH} \
  --few_step true \
  --num_inference_steps 4 \
  --model_type 'ar' \
  --use_vae_parallel false \
  --use_sageattn false \
  --use_fp8_gemm false \
  --transformer_resident_ar_rollout true
