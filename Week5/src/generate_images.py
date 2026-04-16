import os
import pandas as pd
import torch
from diffusers import DiffusionPipeline, DDIMScheduler

def main():
    model_id = "stabilityai/sdxl-turbo"
    out_dir = "../visualizations/generated_clusters2S0CFG"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {model_id}...")
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.enable_sequential_cpu_offload()

    # Set the DDIMScheduler as requested
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Read the CSV
    csv_path = "/ghome/group01/C5/benet/C5-Team1/Week5/embeddings/cleaned_captions.csv"
    print(f"Reading captions from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("Starting generation...")
    # Iterate through captions
    for idx, row in df.iterrows():
        cluster_id = row['cluster']
        caption = row['caption']
        
        # Cover the caption in a prompt
        current_prompt = f"An unedited, amateur smartphone snapshot of {caption}. Casual, everyday indoor lighting, slightly off-center framing, messy background."
        
        print(f"Cluster: {cluster_id} | Generating Image {idx + 1}/{len(df)}")
        print(f"Prompt: {current_prompt}")
        
        # Create cluster folder
        cluster_dir = os.path.join(out_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Generate the image without fixing the seed (random each time)
        image = pipe(
            prompt=current_prompt,
            negative_prompt="professional photography, DSLR, 4k, 8k, highly detailed, studio lighting, bokeh, shallow depth of field, perfect composition, centered, artistic, stock photo, watermark",
            num_inference_steps=2, # 4 steps gives the best quality for Turbo!
            guidance_scale=0.0 # 1.0 is optimal for Turbo (disables CFG)
        ).images[0]
        
        # Save image
        save_path = os.path.join(cluster_dir, f"image_{idx}.png")
        image.save(save_path)
        
    print(f"All images generated and saved to {out_dir}")

if __name__ == "__main__":
    main()