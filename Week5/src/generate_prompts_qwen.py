import os
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Ask Qwen LLM prompts from a CSV")
    parser.add_argument("--csv_path", type=str, default="../embeddings/prompts.csv", help="Path to input CSV")
    parser.add_argument("--output_path", type=str, default="../embeddings/outcomes.csv", help="Path to output CSV")
    # Defaulting to Qwen/Qwen2.5-7B-Instruct. If you have "Qwen/Qwen3.5-9B", you can pass it via this argument.
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading input CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    if "prompt" not in df.columns:
        raise ValueError(f"CSV must contain a 'prompt' column. Found: {list(df.columns)}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    outcomes = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        prompt_text = row["prompt"]
        
        # Format as chat for instruct models
        messages = [
            {"role": "user", "content": prompt_text}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs, 
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        outcomes.append(response.strip())
        
    df["outcome"] = outcomes
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Saved outcomes to {args.output_path}")

if __name__ == "__main__":
    main()
