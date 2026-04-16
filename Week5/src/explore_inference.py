import os
import torch
from diffusers import DiffusionPipeline, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler

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

    # Base configuration adapted for VizWiz dataset
    vz_neg_prompt = "blurry, obstructed, finger in frame, bad lighting, out of focus, low quality, deformed, pixelated, cropped, dark, oversaturated, noisy, watermark, unrealistic"

    vizwiz_prompts = {
        "Image_18_OfficeTV": "A single coherent, photorealistic, highly detailed scene that perfectly fits all of these descriptions: An office room has a TV and a table with papers and a bottle of liquid on it. An office or classroom with a TV on a metal AV stand is in the background of a white table with red chair and glass beverage bottle. A frosted white and orange glass bottle standing on a table with a glass and bowl. A flat screen TV is sitting on a utility cart near a table.",
        
        "Image_48_Hulk": "A single coherent, photorealistic, highly detailed scene that perfectly fits all of these descriptions: a Hulk toy action figure on a carpet area. Green action figure of the Incredible Hulk lying on a brown carpet. Green plastic hulk toy that has purple pants on a carpeted floor. A toy of the Hulk wearing purple pants is on the carpet. An incredible hulk action figure laying on brown carpet.",
        
        "Image_91_TowersHouse": "A single coherent, photorealistic, highly detailed scene that perfectly fits all of these descriptions: A house has a car in the driveway and it is sunny outside. a white and tan two story house with a tree in the front yard under a blue sky. The front of two houses with four antenna towers in the background. A two story house with a tree in the yard. A landscape of a neighborhood with a two story house and a tree right in front of it along with a driveway."
    }
    
    seed = 42 # Keeping the seed constant is CRITICAL for comparison!

    def generate(name, current_prompt, scheduler_cls=None, steps=50, cfg=7.5, negative_prompt=None):
        print(f"Generating: {name}")
        
        # Swap the scheduler if requested
        if scheduler_cls:
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        
        # Set manual seed
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate the image
        image = pipe(
            prompt=current_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator
        ).images[0]
        
        image.save(os.path.join(out_dir, f"{name}.png"))

    print("Starting SEQUENTIAL exploration tests...\n")
    
    for prompt_name, prompt_text in vizwiz_prompts.items():
        print(f"\n=======================================================")
        print(f"EXPLORING PROMPT: {prompt_name}")
        print(f"=======================================================")

        # --- PHASE 1: FIND THE BEST SCHEDULER ---
        # Added Euler and DPM++ 2M (The fastest and most popular SDXL continuous schedulers!)
        print(f"=== PHASE 1: Testing Schedulers ({prompt_name}) ===")
        # generate(f"{prompt_name}_Phase1_A_Scheduler_DDIM", prompt_text, scheduler_cls=DDIMScheduler, steps=50, cfg=7.5, negative_prompt=None)
        # generate(f"{prompt_name}_Phase1_B_Scheduler_DDPM", prompt_text, scheduler_cls=DDPMScheduler, steps=50, cfg=7.5, negative_prompt=None)
        # generate(f"{prompt_name}_Phase1_C_Scheduler_Euler", prompt_text, scheduler_cls=EulerDiscreteScheduler, steps=50, cfg=7.5, negative_prompt=None)
        # generate(f"{prompt_name}_Phase1_D_Scheduler_DPMMultistep", prompt_text, scheduler_cls=DPMSolverMultistepScheduler, steps=50, cfg=7.5, negative_prompt=None)
        
        # -> ASSUMED WINNER: DDIMScheduler (User preference)
        best_scheduler = DDIMScheduler
        pipe.scheduler = best_scheduler.from_config(pipe.scheduler.config)

        # --- PHASE 2: FIND THE BEST DENOISING STEPS ---
        # Added extremely low (5) and extremely high (150) step counts to observe limits
        print(f"=== PHASE 2: Testing Steps (Using DDIM) ===")
        # generate(f"{prompt_name}_Phase2_A_Steps_05", prompt_text, steps=5, cfg=7.5, negative_prompt=None)
        # generate(f"{prompt_name}_Phase2_B_Steps_10", prompt_text, steps=10, cfg=7.5, negative_prompt=None)
        # generate(f"{prompt_name}_Phase2_C_Steps_20", prompt_text, steps=20, cfg=7.5, negative_prompt=None)
        # generate(f"{prompt_name}_Phase2_D_Steps_50", prompt_text, steps=50, cfg=7.5, negative_prompt=None)
        generate(f"{prompt_name}_Phase2_H_Steps_75", prompt_text, steps=75, cfg=7.5, negative_prompt=None)
        # generate(f"{prompt_name}_Phase2_E_Steps_100", prompt_text, steps=100, cfg=7.5, negative_prompt=None)
        # generate(f"{prompt_name}_Phase2_F_Steps_150", prompt_text, steps=150, cfg=7.5, negative_prompt=None)
        
        # -> ASSUMED WINNER: 50 Steps (Perfect balance for SDXL with DDIM)
        best_steps = 50

        # --- PHASE 3: FIND THE BEST CFG SCALE ---
        # Added unconditional (1.0) and super-heavy conditioning (10.0, 25.0)
        print(f"=== PHASE 3: Testing CFG (Using DDIM + 50 Steps) ===")
        # generate(f"{prompt_name}_Phase3_A_CFG_1.0", prompt_text, steps=best_steps, cfg=1.0, negative_prompt=None)
        # generate(f"{prompt_name}_Phase3_B_CFG_2.0", prompt_text, steps=best_steps, cfg=2.0, negative_prompt=None)
        # generate(f"{prompt_name}_Phase3_C_CFG_5.0", prompt_text, steps=best_steps, cfg=5.0, negative_prompt=None)
        # generate(f"{prompt_name}_Phase3_D_CFG_7.5", prompt_text, steps=best_steps, cfg=7.5, negative_prompt=None)
        # generate(f"{prompt_name}_Phase3_E_CFG_10.0", prompt_text, steps=best_steps, cfg=10.0, negative_prompt=None)
        # generate(f"{prompt_name}_Phase3_F_CFG_15.0", prompt_text, steps=best_steps, cfg=15.0, negative_prompt=None)
        # generate(f"{prompt_name}_Phase3_G_CFG_25.0", prompt_text, steps=best_steps, cfg=25.0, negative_prompt=None)

        # -> ASSUMED WINNER: 7.5 CFG (Best prompt adherence without burning out colors)
        best_cfg = 7.5

        # --- PHASE 4: NEGATIVE PROMPTING ---
        # Added a "Light" negative prompt to compare against our "Heavy" VizWiz specific prompt
        print(f"=== PHASE 4: Testing Negative Prompts (Using DDIM + 50 Steps + 7.5 CFG) ===")
        # generate(f"{prompt_name}_Phase4_A_Without_Negative", prompt_text, steps=best_steps, cfg=best_cfg, negative_prompt=None)
        # generate(f"{prompt_name}_Phase4_B_Light_Negative", prompt_text, steps=best_steps, cfg=best_cfg, negative_prompt="bad quality, ugly")
        # generate(f"{prompt_name}_Phase4_C_Heavy_Negative", prompt_text, steps=best_steps, cfg=best_cfg, negative_prompt=vz_neg_prompt)
        
        # -> ASSUMED WINNER: With Heavy Negative Prompts (Removes blurry and deformed artifacts)
        best_neg_prompt = vz_neg_prompt

        # --- ULTIMATE CONFIGURATION ---
        print(f"=== FINAL: Generating Ultimate Configuration ({prompt_name}) ===")
        generate(
            f"{prompt_name}_ULTIMATE_CONFIGURATION", 
            current_prompt=prompt_text,
            scheduler_cls=best_scheduler, 
            steps=best_steps, 
            cfg=best_cfg, 
            negative_prompt=best_neg_prompt
        )
        
    print("\nSequential Exploration complete! Check Week5/visualizations/exploration/ for the results.")

if __name__ == "__main__":
    main()
