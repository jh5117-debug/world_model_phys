import os
import re
import csv
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from template import *

parser = argparse.ArgumentParser()

parser.add_argument('--input_csv', type = str, required = True, help = 'csv')
parser.add_argument('--task', type = str, default='sa', choices=['sa', 'pc', 'rule'], help='task')
parser.add_argument('--checkpoint', type = str, required = True, help = 'checkpoint')
parser.add_argument('--lora_checkpoint', default = None, type = str, help = 'lora trained ckpt')
parser.add_argument('--batch_size', type = int, default = 1)
parser.add_argument('--num_frames', type = int, default = 32)
parser.add_argument('--output_csv', type = str, required = True, help = 'csv')

args = parser.parse_args()


def resolve_model_dtype():
    value = os.environ.get("VIDEOPHY_TORCH_DTYPE", "bfloat16").strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(
            f"Unsupported VIDEOPHY_TORCH_DTYPE={value}. "
            "Use one of: bfloat16, bf16, float16, fp16, float32, fp32."
        )
    return mapping[value]


def _read_bool_env(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_generate_kwargs(model):
    cfg = getattr(model, "generation_config", None)
    kwargs = {
        # Deterministic decoding is much more stable for scoring/eval than the
        # model's chat-style sampling defaults.
        "do_sample": False,
        "pad_token_id": getattr(cfg, "pad_token_id", 0),
        "bos_token_id": getattr(cfg, "bos_token_id", 1),
        "eos_token_id": getattr(cfg, "eos_token_id", 2),
    }

    if "VIDEOPHY_DO_SAMPLE" in os.environ:
        kwargs["do_sample"] = _read_bool_env("VIDEOPHY_DO_SAMPLE", kwargs["do_sample"])

    if kwargs["do_sample"]:
        kwargs["top_k"] = int(
            os.environ.get(
                "VIDEOPHY_TOP_K",
                getattr(cfg, "top_k", 3),
            )
        )
        if "VIDEOPHY_TEMPERATURE" in os.environ:
            kwargs["temperature"] = float(os.environ["VIDEOPHY_TEMPERATURE"])
        elif getattr(cfg, "temperature", None) is not None:
            kwargs["temperature"] = cfg.temperature

    if "VIDEOPHY_MAX_LENGTH" in os.environ:
        kwargs["max_length"] = int(os.environ["VIDEOPHY_MAX_LENGTH"])
    elif getattr(cfg, "max_new_tokens", None) is not None:
        kwargs["max_new_tokens"] = int(cfg.max_new_tokens)
    else:
        # The processor needs generation room up front; leaving this at zero
        # makes the model emit only prompt-aligned tokens on some setups.
        kwargs["max_new_tokens"] = int(os.environ.get("VIDEOPHY_MAX_NEW_TOKENS", "8"))

    if "max_new_tokens" in kwargs:
        kwargs.pop("max_length", None)

    return kwargs

def parse_score_from_output(output):
    text = output.lower().strip()
    if not text:
        return None

    word_map = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    for word, score in word_map.items():
        if re.search(rf"\b{word}\b", text):
            return score

    digit_match = re.search(r"[0-5]", text)
    if digit_match:
        return int(digit_match.group(0))

    return None

def inference(args, model, df, processor, tokenizer, generate_kwargs):
    with torch.no_grad():
        for i,row in tqdm(df.iterrows()):
            videopaths = [row['videopath']]
            if args.task == 'sa':
                prompts = [PROMPT_SA.format(caption=row['caption'])] 
            elif args.task == 'pc':
                prompts = [PROMPT_PHYSICS]
            else:
                prompts = [PROMPT_RULE.format(rule=row['rule'])]
            inputs = processor(text=prompts, videos=videopaths, num_frames=args.num_frames, return_tensors='pt')
            model_dtype = next(model.parameters()).dtype
            inputs = {k: v.to(dtype=model_dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            res = model.generate(**inputs, **generate_kwargs)
            token_ids = res.tolist()[0]
            if "max_new_tokens" in generate_kwargs:
                tail_token_ids = token_ids[-generate_kwargs["max_new_tokens"]:]
            else:
                tail_token_ids = token_ids
            output = tokenizer.decode(tail_token_ids, skip_special_tokens=True)
            output_lower = output.lower().strip()
            
            print(f"[RAW_OUTPUT] {repr(output)}")
            score = parse_score_from_output(output)
            
            if score is None:
                digits = ''.join([c for c in output_lower if c.isdigit()])
                score = int(digits[0]) if digits and digits[0] in "012345" else 0
                print(f"Warning: Could not parse output {repr(output)}. Defaulting to {score}.")
            
            # Set the parsed score as an integer in the dataframe.
            df.at[i, "score"] = score
    return df


def modify_keys(state_dict):
    new_state_dict = defaultdict()

    pattern = re.compile(r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj).weight')

    for key, value in state_dict.items():
        if pattern.match(key):
            key = key.split('.')
            key.insert(-1, 'base_layer')
            key = '.'.join(key)
        new_state_dict[key] = value

    return new_state_dict

def main():

    checkpoint = args.checkpoint
    model_dtype = resolve_model_dtype()

    # Processors
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    df = pd.read_csv(args.input_csv)
    # df = df.iloc[:20]

    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    # Instantiate model
    model = MplugOwlForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=model_dtype,
        device_map={'': 'cpu'}
    )
    print('Model Loaded')
    print(f"Using torch dtype: {model_dtype}")
    model.eval()

    lora_checkpoint = args.lora_checkpoint
    if lora_checkpoint:
        peft_config = LoraConfig(
            target_modules=r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj)', 
            inference_mode=True, 
            r=32, 
            lora_alpha=32, 
            lora_dropout=0.05
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        with open(lora_checkpoint, 'rb') as f:
            ckpt = torch.load(f, map_location = torch.device("cpu"))
        try:
            model.load_state_dict(ckpt)
        except:
            ckpt = modify_keys(ckpt)
            model.load_state_dict(ckpt)
        print("LOADED")
    model = model.to("cuda").to(model_dtype)
    generate_kwargs = build_generate_kwargs(model)
    processor.tokens_to_generate = generate_kwargs.get(
        "max_new_tokens",
        int(os.environ.get("VIDEOPHY_TOKENS_TO_GENERATE", "8")),
    )
    print(f"Processor tokens_to_generate: {processor.tokens_to_generate}")
    print(f"Using generate kwargs: {generate_kwargs}")
    
    out = inference(args, model, df, processor, tokenizer, generate_kwargs)
    out.to_csv(args.output_csv)

if __name__  == "__main__":
    main()

'''
    CUDA_VISIBLE_DEVICES=0 python inference.py --input_csv /local/hbansal/videophy2/human_expts/cogvideox_5b_3x_videophy2_hard_pre_human_annotation_w_bad_sa_pc_eval.csv --checkpoint /local2/hbansal/videophy2/test_videophy_training/videophy_autoeval_three_models_rule_e3_lr5e-4_bs64_part2_vta_pc_rule/videophy_2_autoeval --output_csv /local/hbansal/videophy2/human_expts/cogvideox_5b_3x_videophy2_hard_pre_human_annotation_w_bad_sa_pc_eval_lr5e-4_bs64_part2_vta_pc_rule_502_rerun.csv
'''
