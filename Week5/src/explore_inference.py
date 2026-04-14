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

    print("Starting SEQUENTIAL exploration tests...\n")
    
    # --- PHASE 1: FIND THE BEST SCHEDULER ---
    # Everything else is standard.
    print("=== PHASE 1: Testing Schedulers ===")
    generate("Phase1_A_Scheduler_DDIM", scheduler_cls=DDIMScheduler, steps=50, cfg=7.5, negative_prompt=None)
    generate("Phase1_B_Scheduler_DDPM", scheduler_cls=DDPMScheduler, steps=50, cfg=7.5, negative_prompt=None)
    
    # -> ASSUMED WINNER: DDIMScheduler (Fast, deterministic convergence)
    best_scheduler = DDIMScheduler
    pipe.scheduler = best_scheduler.from_config(pipe.scheduler.config)

    # --- PHASE 2: FIND THE BEST DENOISING STEPS ---
    # Using the best scheduler from Phase 1.
    print("=== PHASE 2: Testing Steps (Using DDIM) ===")
    generate("Phase2_A_Steps_10", steps=10, cfg=7.5, negative_prompt=None)
    generate("Phase2_B_Steps_20", steps=20, cfg=7.5, negative_prompt=None)
    generate("Phase2_C_Steps_50", steps=50, cfg=7.5, negative_prompt=None)
    generate("Phase2_D_Steps_100", steps=100, cfg=7.5, negative_prompt=None)
    
    # -> ASSUMED WINNER: 50 Steps (Best balance of quality vs compute time)
    best_steps = 50

    # --- PHASE 3: FIND THE BEST CFG SCALE ---
    # Using the best scheduler and optimal steps from Phase 1 & 2.
    print("=== PHASE 3: Testing CFG (Using DDIM + 50 Steps) ===")
    generate("Phase3_A_CFG_2.0", steps=best_steps, cfg=2.0, negative_prompt=None)
    generate("Phase3_B_CFG_7.5", steps=best_steps, cfg=7.5, negative_prompt=None)
    generate("Phase3_C_CFG_15.0", steps=best_steps, cfg=15.0, negative_prompt=None)

    # -> ASSUMED WINNER: 7.5 CFG (Best prompt adherence without burning out colors)
    best_cfg = 7.5

    # --- PHASE 4: NEGATIVE PROMPTING ---
    # Using the optimized pipeline from Phases 1, 2, and 3.
    print("=== PHASE 4: Testing Negative Prompts (Using DDIM + 50 Steps + 7.5 CFG) ===")
    generate("Phase4_A_Without_Negative", steps=best_steps, cfg=best_cfg, negative_prompt=None)
    generate("Phase4_B_With_Negative", steps=best_steps, cfg=best_cfg, negative_prompt=neg_prompt)
    
    # -> ASSUMED WINNER: With Negative Prompts (Removes blurry and deformed artifacts)
    best_neg_prompt = neg_prompt

    # --- ULTIMATE CONFIGURATION ---
    print("=== FINAL: Generating Ultimate Configuration ===")
    generate(
        "ULTIMATE_CONFIGURATION", 
        scheduler_cls=best_scheduler, 
        steps=best_steps, 
        cfg=best_cfg, 
        negative_prompt=best_neg_prompt
    )
    
    print("\nSequential Exploration complete! Check Week5/visualizations/exploration/ for the results.")

if __name__ == "__main__":
    main()
