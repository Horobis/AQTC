export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960
export VIDEO_READER_BACKEND=torchvision
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#export TORCHCODEC_DISABLE=1
export FORCE_QWENVL_VIDEO_READER=decord
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

accelerate launch --num_processes 1 --gpu_ids 0 --main_process_port 8000 -m lmms_eval \
    --model qwen3_vl_aqtc \
    --force_simple \
    --model_args pretrained=/mnt/data_A/zhuyan/HugModel/Qwen/Qwen3-VL-8B-Instruct,keep_ratio=0.25,novelty_frames_per_segment=4,query_weight=0.5,max_pixels=151200,min_pixels=143360,max_num_frames=256,dynamic_segment_budget=0,dynamic_budget_strength=0.7,budget_ceiling=0.55,temporal_aware=0,temporal_weight=0.3,temporal_protect_ratio=0.35 \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen3_vl_8b_longvideobench \
    --output_path /mnt/data_A/zhuyan/Results/AQTC/result/aqtc/longvideobench/