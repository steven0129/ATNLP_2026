import re

def sft_formatting_prompts_func(tokenizer, example):
    q = example["question"].strip()
    raw_answer = example["answer"].strip()
    
    if "####" in raw_answer:
        reasoning, final_ans = raw_answer.split("####", 1)
        reasoning = reasoning.strip()
        final_ans = final_ans.strip()
    else:
        reasoning = raw_answer
        final_ans = ""

    formatted_answer = f"{reasoning}\n\nThe answer is {final_ans}"

    instruction = (
        "Please think step by step before answering the question, and provide the final answer as 'the answer is [answer]' format. "
        "These problems take between 2 and 8 steps to solve. "
        "Solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer. "
        "A bright middle school student should be able to solve every problem: from the paper, 'Problems require no concepts beyond the level of early Algebra, and the vast majority of problems can be solved without explicitly defining a variable.' "
        "Solutions are provided in natural language, as opposed to pure math expressions."
    )

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": q},
        {"role": "assistant", "content": formatted_answer}
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    formatted_text += tokenizer.eos_token
    
    return {"text": formatted_text}
