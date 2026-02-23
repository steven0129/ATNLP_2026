import re

def build_rl_dataset(example):
    
    system_prompt = (
        "Please think step by step before answering the question, and provide the final answer as 'the answer is [answer]' format. "
        "These problems take between 2 and 8 steps to solve. "
        "Solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer. "
        "A bright middle school student should be able to solve every problem: from the paper, 'Problems require no concepts beyond the level of early Algebra, and the vast majority of problems can be solved without explicitly defining a variable.' "
        "Solutions are provided in natural language, as opposed to pure math expressions."
    )

    truth = example['answer'].split("####")[1].strip()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['question']}
    ]
    
    return {
        "prompt": messages,
        "answer": truth
    }
    