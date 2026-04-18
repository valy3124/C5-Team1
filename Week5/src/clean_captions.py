import os
import argparse
import pandas as pd
import re
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Clean generated captions from outcomes CSV")
    parser.add_argument("--csv_path", type=str, default="../embeddings/outcomes.csv", help="Path to input outcomes CSV")
    parser.add_argument("--output_path", type=str, default="../embeddings/cleaned_captions.csv", help="Path to output cleaned CSV")
    return parser.parse_args()

def clean_outcome(text):
    captions = []
    if not isinstance(text, str):
        return captions
        
    for line in text.split('\n'):
        line = line.strip()
        # Look for lines starting with a number and a dot (e.g. "1. ")
        match = re.match(r'^\d+\.\s+(.*)', line)
        if match:
            caption = match.group(1)
            
            # Remove leading bolded categories if present (e.g., "**Blurry Prompt**: ")
            caption = re.sub(r'^\*\*.*?\*\*\s*:\s*', '', caption)
            
            # Remove any remaining asterisks used for markdown formatting
            caption = caption.replace('**', '').replace('*', '')
            
            # Strip remaining whitespace and quotes
            caption = caption.strip().strip('"').strip("'").strip()
            
            if caption:
                captions.append(caption)
                
    return captions

def main():
    args = parse_args()
    
    print(f"Loading input CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    cluster_col = "cluster_id" if "cluster_id" in df.columns else "cluster"
    
    if "outcome" not in df.columns or cluster_col not in df.columns:
        raise ValueError(f"CSV must contain '{cluster_col}' and 'outcome' columns. Found columns: {list(df.columns)}")
    
    cleaned_rows = []
    
    for _, row in df.iterrows():
        cluster_val = row[cluster_col]
        outcome_text = row['outcome']
        
        # Extract and clean captions from the LLM's text block
        captions = clean_outcome(outcome_text)
        
        if "num_samples_to_generate" in df.columns:
            target_count = int(row["num_samples_to_generate"])
            if len(captions) > target_count:
                # Randomly drop the extra captions
                captions = random.sample(captions, target_count)
            elif len(captions) < target_count and len(captions) > 0:
                # Randomly replicate some captions to fill the gap
                shortfall = target_count - len(captions)
                extra_captions = random.choices(captions, k=shortfall)
                captions.extend(extra_captions)
        
        # Add a new row for each newly extracted caption
        for cap in captions:
            cleaned_rows.append({
                'cluster': cluster_val,
                'caption': cap
            })
            
    out_df = pd.DataFrame(cleaned_rows)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    
    print(f"Successfully processed {len(df)} outcomes into {len(out_df)} cleaned individual captions!")
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()
