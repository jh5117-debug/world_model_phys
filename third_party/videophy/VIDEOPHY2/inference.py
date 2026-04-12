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
from output_parsing import parse_score_from_output
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

_SCORE_TO_WORD = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
}


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
        # Stay close to the upstream inference defaults unless explicitly
        # overridden. This keeps eval behavior comparable across environments.
        "do_sample": False,
        "top_k": 1,
        "temperature": 0.001,
        "pad_token_id": getattr(cfg, "pad_token_id", 0),
        "bos_token_id": getattr(cfg, "bos_token_id", 1),
        "eos_token_id": getattr(cfg, "eos_token_id", 2),
    }

    if "VIDEOPHY_DO_SAMPLE" in os.environ:
        kwargs["do_sample"] = _read_bool_env("VIDEOPHY_DO_SAMPLE", kwargs["do_sample"])

    if "VIDEOPHY_TOP_K" in os.environ:
        kwargs["top_k"] = int(os.environ["VIDEOPHY_TOP_K"])
    elif getattr(cfg, "top_k", None) is not None:
        kwargs["top_k"] = int(cfg.top_k)

    if "VIDEOPHY_TEMPERATURE" in os.environ:
        kwargs["temperature"] = float(os.environ["VIDEOPHY_TEMPERATURE"])
    elif getattr(cfg, "temperature", None) is not None:
        kwargs["temperature"] = float(cfg.temperature)

    if "VIDEOPHY_MAX_NEW_TOKENS" in os.environ:
        kwargs["max_new_tokens"] = int(os.environ["VIDEOPHY_MAX_NEW_TOKENS"])
    elif "VIDEOPHY_MAX_LENGTH" in os.environ:
        kwargs["max_length"] = int(os.environ["VIDEOPHY_MAX_LENGTH"])
    else:
        # Score-only prompts should finish within a couple of tokens. Keeping
        # the generation short avoids garbage tails like "100000".
        kwargs["max_new_tokens"] = 4

    if "max_new_tokens" in kwargs:
        kwargs.pop("max_length", None)

    return kwargs


def resolve_tokens_to_generate():
    raw = os.environ.get("VIDEOPHY_TOKENS_TO_GENERATE")
    if raw is None:
        return 0
    return int(raw)


def _prepare_model_inputs(inputs, *, model, model_dtype):
    prepared = {}
    for key, value in inputs.items():
        if key == "attention_mask":
            prepared[key] = value.to(device=model.device, dtype=torch.long)
        elif value.dtype == torch.float:
            prepared[key] = value.to(device=model.device, dtype=model_dtype)
        else:
            prepared[key] = value.to(model.device)
    return prepared


def _allowed_scores_for_task(task):
    if task == "rule":
        return (0, 1, 2)
    return (1, 2, 3, 4, 5)


def _build_score_token_map(tokenizer, task):
    token_map = {}
    for score in _allowed_scores_for_task(task):
        token_ids = set()
        for variant in (
            str(score),
            f" {score}",
            f"\n{score}",
            _SCORE_TO_WORD[score],
            f" {_SCORE_TO_WORD[score]}",
            f"\n{_SCORE_TO_WORD[score]}",
        ):
            encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if len(encoded) == 1:
                token_ids.add(int(encoded[0]))
        if token_ids:
            token_map[score] = sorted(token_ids)
    return token_map


def _score_from_next_token_logits(next_token_logits, score_token_map):
    if not score_token_map:
        return None, None

    allowed_scores = []
    allowed_logits = []
    for score in sorted(score_token_map):
        token_ids = score_token_map[score]
        score_logit = max(float(next_token_logits[token_id].item()) for token_id in token_ids)
        allowed_scores.append(score)
        allowed_logits.append(score_logit)

    logits_tensor = torch.tensor(allowed_logits, dtype=torch.float32)
    probs = torch.softmax(logits_tensor, dim=0)
    best_idx = int(torch.argmax(probs).item())
    return allowed_scores[best_idx], float(probs[best_idx].item())


def _decode_generation_outputs(sequences, *, prompt_token_count, tokenizer):
    token_ids = sequences[0].tolist()
    generated_token_ids = token_ids[prompt_token_count:]
    full_output = tokenizer.decode(token_ids, skip_special_tokens=True)
    generated_output = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return full_output, generated_output


def inference(args, model, df, processor, tokenizer, generate_kwargs, score_token_map):
    with torch.no_grad():
        for i, row in tqdm(df.iterrows()):
            videopaths = [row['videopath']]
            if args.task == 'sa':
                prompts = [PROMPT_SA.format(caption=row['caption'])] 
            elif args.task == 'pc':
                prompts = [PROMPT_PHYSICS]
            else:
                prompts = [PROMPT_RULE.format(rule=row['rule'])]
            inputs = processor(text=prompts, videos=videopaths, num_frames=args.num_frames, return_tensors='pt')
            model_dtype = next(model.parameters()).dtype
            inputs = _prepare_model_inputs(inputs, model=model, model_dtype=model_dtype)

            generate_call_kwargs = dict(generate_kwargs)
            generate_call_kwargs["return_dict_in_generate"] = True
            generate_call_kwargs["output_scores"] = True

            res = model.generate(**inputs, **generate_call_kwargs)
            sequences = res.sequences if hasattr(res, "sequences") else res
            choice_score = None
            choice_confidence = None
            if hasattr(res, "scores") and res.scores:
                choice_score, choice_confidence = _score_from_next_token_logits(
                    res.scores[0][0],
                    score_token_map,
                )
            prompt_token_count = int(inputs["input_ids"].shape[1])
            full_output, generated_output = _decode_generation_outputs(
                sequences,
                prompt_token_count=prompt_token_count,
                tokenizer=tokenizer,
            )

            print(f"[RAW_OUTPUT] {repr(full_output)}")
            print(f"[GENERATED_OUTPUT] {repr(generated_output)}")
            if choice_score is not None:
                print(
                    f"[CHOICE_SCORE] score={choice_score} confidence={choice_confidence:.4f}"
                )

            score = parse_score_from_output(generated_output, task=args.task)
            score_source = "generated_output"
            if score is None:
                score = parse_score_from_output(full_output, task=args.task)
                if score is not None:
                    score_source = "full_output"
            if score is None:
                score = choice_score
                score_source = "choice_logits" if choice_score is not None else ""
                print(
                    f"Warning: Could not parse generated output {repr(generated_output)}. "
                    f"Falling back to {score_source or 'empty score'}."
                )

            df.at[i, "raw_output"] = full_output
            df.at[i, "generated_output"] = generated_output
            df.at[i, "choice_score"] = choice_score
            df.at[i, "choice_confidence"] = choice_confidence
            df.at[i, "score_source"] = score_source
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
    processor.tokens_to_generate = resolve_tokens_to_generate()
    score_token_map = _build_score_token_map(tokenizer, args.task)
    print(f"Processor tokens_to_generate: {processor.tokens_to_generate}")
    print(f"Using generate kwargs: {generate_kwargs}")
    print(f"Score token map: {score_token_map}")
    
    out = inference(args, model, df, processor, tokenizer, generate_kwargs, score_token_map)
    out.to_csv(args.output_csv, index=False)

if __name__  == "__main__":
    main()

'''
    CUDA_VISIBLE_DEVICES=0 python inference.py --input_csv /local/hbansal/videophy2/human_expts/cogvideox_5b_3x_videophy2_hard_pre_human_annotation_w_bad_sa_pc_eval.csv --checkpoint /local2/hbansal/videophy2/test_videophy_training/videophy_autoeval_three_models_rule_e3_lr5e-4_bs64_part2_vta_pc_rule/videophy_2_autoeval --output_csv /local/hbansal/videophy2/human_expts/cogvideox_5b_3x_videophy2_hard_pre_human_annotation_w_bad_sa_pc_eval_lr5e-4_bs64_part2_vta_pc_rule_502_rerun.csv
'''
