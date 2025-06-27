kill -9 $(lsof -t /dev/nvidia*)
sleep 1s
kill -9 $(lsof -t /dev/nvidia*)


NUM_STEPS=50
CFG_SCALE=4.0
TIME_SHIFTING_FACTOR=6
SEED=20
SOLVER=euler
TASK_TYPE="Image Infilling"
CAP_DIR=./examples/caption_list.json
OUT_DIR=./examples/outputs
MODEL_CHECKPOINT=/root/paddlejob/workspace/env_run/output/lvfeng/model/models--Alpha-VLLM--Lumina-Accessory/snapshots/711d5d6656c62957e8625b02ea53cc74f2c5589d

python -u sample_accessory.py --ckpt ${MODEL_CHECKPOINT} \
--image_save_path ${OUT_DIR} \
--solver ${SOLVER} \
--num_sampling_steps ${NUM_STEPS} \
--caption_path ${CAP_DIR} \
--seed ${SEED} \
--time_shifting_factor ${TIME_SHIFTING_FACTOR} \
--cfg_scale ${CFG_SCALE} \
--batch_size 1 \
--rank 0 \
--task_type "${TASK_TYPE}"