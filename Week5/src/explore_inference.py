import os
import torch
from diffusers import DiffusionPipeline, DDIMScheduler, DDPMScheduler

def main():
    # We will use SDXL as our base for exploration
    # It is from StabilityAI and is a perfect discrete diffusion model for Tasks like DDPM vs DDIM, CFG, etc.
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    out_dir = "../visualizations/exploration"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {model_id}...")
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    # We must use sequential CPU offload for SDXL to prevent OOM
    pipe.enable_sequential_cpu_offload()

    # Base configuration
    prompt = "A majestic dragon flying over a medieval castle, highly detailed, digital art"
    neg_prompt = "blurry, low quality, deformed, pixelated, ugly"
    seed = 42 # Keeping the seed constant is CRITICAL for comparison!

    def generate(name, scheduler_cls=None, steps=50, cfg=7.5, negative_prompt=None):
        print(f"Generating: {name}")
        
        # Swap the scheduler if requested
        if scheduler_cls:
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        
        # Set manual seed
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate the image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator
        ).images[0]
        
        image.save(os.path.join(out_dir, f"{name}.png"))

    print("Starting exploration tests...\n")
    
    # --- EXPERIMENT 1: SCHEDULERS ---
    print("=== Testing Schedulers ===")
    generate("01_Scheduler_DDIM", DDIMScheduler)
    generate("02_Scheduler_DDPM", DDPMScheduler)
    
    # Restore to DDIM for the rest of the tests for speed
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # --- EXPERIMENT 2: DENOISING STEPS ---
    print("=== Testing Steps ===")
    generate("03_Steps_10", steps=10)
    generate("04_Steps_20", steps=20)
    generate("05_Steps_50", steps=50) # Same as baseline
    generate("06_Steps_100", steps=100)

    # --- EXPERIMENT 3: GUIDANCE SCALE (CFG) ---
    print("=== Testing CFG ===")
    generate("07_CFG_2.0", cfg=2.0)
    generate("08_CFG_7.5", cfg=7.5) # Same as baseline
    generate("09_CFG_15.0", cfg=15.0)

    # --- EXPERIMENT 4: POSITIVE VS NEGATIVE PROMPTING ---
    print("=== Testing Negative Prompts ===")
    generate("10_Prompt_Without_Negative")
    generate("11_Prompt_With_Negative", negative_prompt=neg_prompt)
    
    print("\nExploration complete! Check Week5/visualizations/exploration/ for the results.")

if __name__ == "__main__":
    main()
