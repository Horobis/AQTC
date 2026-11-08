"""LLaVA-Video AQTC local demo. Offline / GPU env MUST be set before importing torch/llava."""
import os
import sys
from pathlib import Path

# Prefer AQTC vendored llava under aqtc_models/
AQTC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AQTC_ROOT / "aqtc_models"))

# Prevent conversation.py from retrying to download Meta-Llama-3 tokenizer.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import copy
import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
import llava

pretrained = "/mnt/data_A/zhuyan/HugModel/LLaVA-Video-7B-Qwen2"
video_dir = Path("/mnt/data_A/zhuyan/DataSet/frame_data/videomme/L_fFjv93ACGo8")
desc = "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?。"
conv_template = "qwen_1_5"
# Match VideoMME profiling (64 frames, keep_ratio=0.50). Change back after timing.
max_frames = 64

# AQTC controls (aligned with Qwen demo_model.py semantics)
keep_ratio = 0.30
memory_capacity = 256
query_weight = 0.1
saliency_weight = 0.9
novelty_frames_per_segment = 4
dynamic_segment_budget = 1
dynamic_budget_strength = 0.3
budget_ceiling = 0.55

device = torch.device("cuda:6")
device_map = "cuda:7"

image_paths = sorted(video_dir.glob("*.jpg"), key=lambda x: int(x.stem))
frames = [Image.open(p).convert("RGB") for p in image_paths[:max_frames]]
assert frames, f"No jpg frames under {video_dir}"

print("llava package:", llava.__file__)
model_name = get_model_name_from_path(pretrained)
tokenizer, model, image_processor, _ = load_pretrained_model(
    pretrained,
    None,
    model_name,
    device_map=device_map,
    torch_dtype="bfloat16",
    attn_implementation="sdpa",
    multimodal=True,
    overwrite_config={
        "mm_spatial_pool_stride": 2,
        "mm_spatial_pool_mode": "bilinear",
        "mm_newline_position": "grid",
    },
    keep_ratio=keep_ratio,
    memory_capacity=memory_capacity,
    query_weight=query_weight,
    saliency_weight=saliency_weight,
    novelty_frames_per_segment=novelty_frames_per_segment,
    dynamic_segment_budget=dynamic_segment_budget,
    dynamic_budget_strength=dynamic_budget_strength,
    budget_ceiling=budget_ceiling,
    aqtc_enabled=True,
)
model.eval()
model.to(device=device)
model.reset_novelty_memory()
model.reset_token_stats()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

video = image_processor.preprocess(
    [np.array(f) for f in frames],
    return_tensors="pt",
)["pixel_values"].to(device=device, dtype=torch.bfloat16)
video = [video]

question = DEFAULT_IMAGE_TOKEN + f"\n{desc}"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(device)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=512,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize(device)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"[AQTC] peak CUDA memory: {peak_gb:.2f} GB")

stats = model.get_token_stats()
orig = stats["original_visual"]
comp = stats["compressed_visual"]
ratio = (comp / orig) if orig else float("nan")
print(
    f"[AQTC] visual tokens: {orig} -> {comp} "
    f"(keep_ratio={keep_ratio}, empirical={ratio:.4f})"
)

text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print("---------------")
print(text)
