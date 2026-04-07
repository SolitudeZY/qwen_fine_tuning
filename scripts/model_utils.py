"""
统一模型加载工具 —— 同时支持 Qwen2.5-VL 和 Qwen3-VL 系列

用法:
    from model_utils import load_vlm, infer_vlm

    model, processor, model_family = load_vlm(model_path, lora_path=None)
    response_text = infer_vlm(model, processor, model_family, messages, max_new_tokens=512)

model_family 取值:
    "qwen2.5-vl"  → Qwen2_5_VLForConditionalGeneration
    "qwen3-vl"    → AutoModelForImageTextToText
"""

import json
import os
from typing import Optional

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig


def detect_model_family(model_path: str) -> str:
    """根据模型路径或 config.json 自动检测模型系列"""
    path_lower = model_path.lower()

    # 先按路径名判断
    if "qwen3-vl" in path_lower or "qwen3_vl" in path_lower:
        return "qwen3-vl"
    if "qwen2.5-vl" in path_lower or "qwen2_5-vl" in path_lower or "qwen2___5" in path_lower:
        return "qwen2.5-vl"

    # 再读 config.json 判断
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model_type = config.get("model_type", "").lower()
        architectures = [a.lower() for a in config.get("architectures", [])]

        if "qwen3_vl" in model_type or any("qwen3" in a for a in architectures):
            return "qwen3-vl"
        if "qwen2_5_vl" in model_type or any("qwen2_5" in a or "qwen2.5" in a for a in architectures):
            return "qwen2.5-vl"
        # Qwen2-VL 也走 qwen2.5-vl 分支（API 兼容）
        if "qwen2_vl" in model_type:
            return "qwen2.5-vl"

    # 默认回退
    print(f"[model_utils] 无法自动检测模型系列，默认使用 qwen3-vl: {model_path}")
    return "qwen3-vl"


def load_vlm(
    model_path: str,
    lora_path: Optional[str] = None,
    use_4bit: bool = True,
    device_map: str = "auto",
) -> tuple:
    """
    统一加载 VLM 模型。

    返回: (model, processor, model_family)
    """
    family = detect_model_family(model_path)
    print(f"[model_utils] 检测到模型系列: {family}")
    print(f"[model_utils] 加载基础模型: {model_path}")

    # 量化配置（对 2B/4B 小模型可选，但仍推荐节省显存）
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    if family == "qwen3-vl":
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=quant_config,
        )
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=quant_config,
        )

    if lora_path and os.path.exists(lora_path):
        print(f"[model_utils] 加载 LoRA 权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    model.eval()

    return model, processor, family


def infer_vlm(
    model,
    processor,
    model_family: str,
    messages: list,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> str:
    """
    统一推理接口。

    对 Qwen3-VL 自动关闭 thinking 模式（/no_think）以提升速度。
    返回模型生成的文本。
    """
    # Qwen3-VL: 关闭 thinking 模式提升速度
    enable_thinking = False if model_family == "qwen3-vl" else None

    chat_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = enable_thinking

    text = processor.apply_chat_template(messages, **chat_kwargs)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if do_sample:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    generated = [
        out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response
