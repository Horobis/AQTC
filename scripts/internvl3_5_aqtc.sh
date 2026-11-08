export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


accelerate launch --num_processes 1 --gpu_ids 1 --main_process_port 8000 -m lmms_eval \
    --model internvl3_5_aqtc \
    --model_args pretrained=/mnt/data_A/zhuyan/HugModel/OpenGVLab/InternVL3_5-8B,modality=video,num_frame=64,vision_chunk_frames=4,use_flash_attn=False,llm_attn_implementation=sdpa,keep_ratio=1.0,novelty_frames_per_segment=4,query_weight=0.5,dynamic_segment_budget=0,dynamic_budget_strength=0.7,budget_ceiling=0.45,temporal_aware=0,temporal_weight=0.6,temporal_protect_ratio=0.35 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix internvl3_5_videomme \
    --output_path /mnt/data_A/zhuyan/Results/AQTC/result/baseline/videomme/
