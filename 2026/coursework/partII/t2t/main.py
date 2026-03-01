import argparse
import json
import os
import random
import re

import numpy as np
import torch
import wandb
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from dataset import build_rl_dataset


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def _extract_completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return completion.get("content", "")
    if isinstance(completion, (list, tuple)):
        parts = []
        for item in completion:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("content", ""))
        return "".join(parts)
    return ""


def _prompt_key(prompt):
    if isinstance(prompt, (list, dict)):
        try:
            return json.dumps(prompt, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(prompt)
    return str(prompt)


def make_t2t_reward_func(tokenizer, alpha, length_min, length_max):
    pattern = re.compile(r"The answer is[:\s]*([^\.\n]+)", re.IGNORECASE)

    def _length_score(text):
        length = len(tokenizer(text, add_special_tokens=False).input_ids)
        if length_max <= length_min:
            return 0.0
        normalized = (length - length_min) / (length_max - length_min)
        if normalized < 0.0:
            return 0.0
        if normalized > 1.0:
            return 1.0
        return float(normalized)

    def reward_func(prompts, completions, answer, **kwargs):
        correctness = []
        length_scores = []
        prompt_keys = []

        for prompt, completion, gold in zip(prompts, completions, answer):
            text = _extract_completion_text(completion)
            match = pattern.search(text.strip())
            if match:
                predicted = match.group(1).replace(",", "").strip()
                gold_clean = str(gold).replace(",", "").strip()
                is_correct = 1 if predicted == gold_clean else 0
            else:
                is_correct = 0

            correctness.append(is_correct)
            length_scores.append(_length_score(text))
            prompt_keys.append(_prompt_key(prompt))

        pass_rate_sum = {}
        pass_rate_count = {}
        for key, is_correct in zip(prompt_keys, correctness):
            pass_rate_sum[key] = pass_rate_sum.get(key, 0.0) + float(is_correct)
            pass_rate_count[key] = pass_rate_count.get(key, 0) + 1

        rewards = []
        for key, is_correct, length_score in zip(prompt_keys, correctness, length_scores):
            p_hat = pass_rate_sum[key] / pass_rate_count[key]
            if is_correct:
                reward = 1.0 - alpha * length_score * p_hat
            else:
                reward = alpha * length_score * (1.0 - p_hat)
            rewards.append(float(reward))

        return rewards

    return reward_func


def main():
    parser = argparse.ArgumentParser(description="T2T Training Script")
    parser.add_argument("--model_signature", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", required=True, help="Path to the SFT adapter checkpoint")
    parser.add_argument("--output_path", required=True, default="./t2t_output")
    parser.add_argument("--dataset_path", default="dataset/gsm8k_500_grpo")
    parser.add_argument("--wandb_project", default="nlu-gsm8k-t2t")
    parser.add_argument("--wandb_token", required=True, default=None)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--length_min", type=int, default=0)
    parser.add_argument("--length_max", type=int, default=4096)

    args = parser.parse_args()

    if args.wandb_token:
        print(f"Logging into WandB with provided token {args.wandb_token}...")
        wandb.login(key=args.wandb_token)

    run_name = args.output_path.split("/")[-1]

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_signature, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Base Model: {args.model_signature}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_signature,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    print(f"Loading and Merging SFT Adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.merge_and_unload()

    t2t_peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj"],
    )

    dataset = load_from_disk(args.dataset_path)
    train_dataset = dataset.map(build_rl_dataset)

    training_args = GRPOConfig(
        output_dir=args.output_path,
        run_name=run_name,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=256,
        num_train_epochs=1,
        bf16=True,
        logging_steps=5,
        report_to="wandb",
        save_strategy="steps",
        save_steps=100,
        beta=0.1,
    )

    reward_func = make_t2t_reward_func(
        tokenizer=tokenizer,
        alpha=args.alpha,
        length_min=args.length_min,
        length_max=args.length_max,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=t2t_peft_config,
    )

    print("Starting T2T Training...")
    trainer.train()

    print("Saving T2T Adapter...")
    trainer.save_model(args.output_path)
    wandb.finish()


if __name__ == "__main__":
    main()
