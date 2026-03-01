import json
import re
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from peft import PeftModel
from math_verify import parse, verify

# ================= 配置区 =================
current_dir = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER_PATH = os.path.join(current_dir, "checkpoint/Qwen/Qwen2.5-0.5B-Instruct-sft/checkpoint-170")
GRPO_ADAPTER_PATH = os.path.join(current_dir, "checkpoint/Qwen/Qwen2.5-0.5B-Instruct-sft_grpo/checkpoint-125")

DATASET_DIR = os.path.join(current_dir, "dataset/gsm8k_3k_sft")
OUTPUT_PATH = os.path.join(current_dir, "dataset/gsm8k_refined_logic.jsonl")

NUM_GENERATIONS = 16 
BATCH_SIZE = 16       # 显存充足，设为 8 加速
MAX_NEW_TOKENS = 512
LIMIT_ORIGINAL_DATA = 1000  # 🎯 只处理原数据集的前 1000 条
# ==========================================

def extract_answer(text):
    match = re.search(r"the\s+answer\s+is[:\s]*([^\.\n]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def is_reasoning_rich(text):
    ops = len(re.findall(r"[\+\-\*\/=]", text))
    return ops >= 3

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print("Merging Adapters...")
    model = PeftModel.from_pretrained(base_model, os.path.abspath(SFT_ADAPTER_PATH))
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, os.path.abspath(GRPO_ADAPTER_PATH))
    model.eval()

    # 加载数据集并只取前 1000 条
    dataset = load_from_disk(DATASET_DIR).select(range(LIMIT_ORIGINAL_DATA))
    gold_samples = []
    instruction = "Think step by step before answering the question, and provide the final answer as 'the answer is [answer]' format."

    print(f"开始处理前 {LIMIT_ORIGINAL_DATA} 条原始数据...")

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch_dict = dataset[i : i + BATCH_SIZE]
        questions = batch_dict['question']
        answers = batch_dict['answer']
        
        prompts = [tokenizer.apply_chat_template([
            {"role": "system", "content": instruction},
            {"role": "user", "content": q.strip()}
        ], tokenize=False, add_generation_prompt=True) for q in questions]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=NUM_GENERATIONS,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        for b_idx in range(len(questions)):
            candidates = decoded_outputs[b_idx * NUM_GENERATIONS : (b_idx + 1) * NUM_GENERATIONS]
            best_cand = None
            max_len = -1
            
            for cand in candidates:
                if "the answer is" in cand.lower() and is_reasoning_rich(cand):
                    pred = extract_answer(cand)
                    if pred and verify(parse(answers[b_idx]), parse(pred)):
                        if len(cand) > max_len:
                            max_len = len(cand)
                            best_cand = cand
            
            if best_cand:
                gold_samples.append({"question": questions[b_idx], "answer": best_cand})

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sample in gold_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n处理完毕！从 1000 条原数据中成功提取了 {len(gold_samples)} 条黄金样本。")

if __name__ == "__main__":
    main()