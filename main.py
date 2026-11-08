import transformers
from transformers import AutoProcessor
from PIL import Image
from pathlib import Path
import torch
from qwen_vl_utils import process_vision_info

print(transformers.__version__)
from aqtc_models import Qwen3VLForConditionalGeneration
# from aqtc_models.demo_model import Qwen3VLForConditionalGeneration
from transformers import AutoConfig, AutoModel, AutoTokenizer
model_path = "/mnt/data_A/zhuyan/HugModel/Qwen/Qwen3-VL-8B-Instruct"

video_dir = Path("/mnt/data_A/zhuyan/DataSet/frame_data/videomme/L_fFjv93ACGo8")
image_paths = sorted(video_dir.glob("*.jpg"), key=lambda x: int(x.stem))
videos = [Image.open(img_path) for img_path in image_paths]
videos = videos[0:16]
desc = "请描述这个视频。"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype="auto",
    device_map="cuda:4",
    keep_ratio=0.75,
    memory_capacity=256,
    query_weight=0.7,
    saliency_weight=0.3,
    novelty_frames_per_segment=4,
    dynamic_segment_budget=True,
    dynamic_budget_strength=0.7,
    budget_ceiling=0.55,
    temporal_aware=True,
    temporal_weight=0.6,
    temporal_protect_ratio=0.35,
)
# model = AutoModel.from_pretrained(model_path, device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)
min_pixels: int = 256 * 28 * 28
max_pixels: int = 256 * 32 * 32


def video_main():
    """Video mode: pass frame file paths as a video list to processor."""
    messages = [
        {"role": "user", "content": [
            {
                "type": "video",
                "video": [str(p) for p in image_paths[:len(videos)]]
            },
            {"type": "text", "text": desc},
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        do_sample_frames=False
    ).to(model.device)
    inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    inputs["mm_token_type_ids"][inputs["input_ids"] == model.config.video_token_id] = 2
    print(inputs.keys())
    print(torch.unique(inputs["mm_token_type_ids"]))
    print(inputs.get("video_grid_thw"))
    print(inputs.get("image_grid_thw"))
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(response)


def main():
    """Image mode: each frame as an independent image with min/max_pixels control."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img, "max_pixels": max_pixels, "min_pixels": min_pixels}
            for img in videos
        ] + [{"type": "text", "text": desc}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)



    print("model image_token_id:", model.config.image_token_id)
    print("model video_token_id:", model.config.video_token_id)

    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print("processor image_pad_id:", image_pad_id)

    print("input_ids shape:", inputs["input_ids"].shape)
    print(
        "image token count:",
        (inputs["input_ids"] == model.config.image_token_id).sum().item()
    )
    print(
        "processor image token count:",
        (inputs["input_ids"] == image_pad_id).sum().item()
    )
    print("inputs keys:", inputs.keys())


    inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    inputs["mm_token_type_ids"][inputs["input_ids"] == model.config.image_token_id] = 1
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(response)


if __name__ == '__main__':
    # main()
    video_main()
