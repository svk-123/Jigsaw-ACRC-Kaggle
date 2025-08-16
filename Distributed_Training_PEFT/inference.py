import os
os.environ["VLLM_USE_V1"] = "0"

import vllm
import torch
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from datasets import Dataset
from vllm.lora.request import LoRARequest
from get_dataset import build_dataset
from config import BASE_MODEL_PATH, LORA_PATH, DATA_PATH


def main():
    llm = vllm.LLM(
        BASE_MODEL_PATH,
        quantization="gptq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=4096,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=64,
    )
    
    tokenizer = llm.get_tokenizer()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["True", "False"])
    
    test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")
    test_dataset = build_dataset(test_dataframe, tokenizer)  
    
    texts = test_dataset["text"]  
    outputs = llm.generate(
        texts,
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[mclp],
            logprobs=2,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("default", 1, LORA_PATH)
    )
    
    log_probs = [
        {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
        for out in outputs
    ]
    
    # Extract True/False probabilities
    predictions = pd.DataFrame(log_probs)[["True", "False"]]
    predictions["row_id"] = test_dataframe["row_id"]
    
    # Create submission with "True" probability as rule_violation score
    submission = predictions[["row_id", "True"]].rename(columns={"True": "rule_violation"})
    submission.to_csv("submission.csv", index=False)
    
    print(f"Submission saved with {len(submission)} predictions")
    print("Sample predictions:")
    print(submission.head())

if __name__ == "__main__":
    main()