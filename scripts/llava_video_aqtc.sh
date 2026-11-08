export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/mnt/data_A/zhuyan/nltk_data
export DECORD_EOF_RETRY_MAX=40960
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export AQTC_PROFILING=1

accelerate launch --num_processes 1 --gpu_ids 4 --main_process_port 8000 -m lmms_eval \
    --model llava_aqtc \
    --model_args pretrained=/mnt/data_A/zhuyan/HugModel/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=128,mm_spatial_pool_mode=average,attn_implementation=sdpa,keep_ratio=0.5,novelty_frames_per_segment=4,query_weight=0.5,dynamic_segment_budget=0,dynamic_budget_strength=0.7,budget_ceiling=0.45,temporal_aware=0,temporal_weight=0.3,temporal_protect_ratio=0.45 \
    --tasks mlvu_dev \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_video_7b_mlvu \
    --output_path /mnt/data_A/zhuyan/Results/AQTC/result/aqtc/mlvu/
