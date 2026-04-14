import torch
import argparse
import os
from diffusers import DiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion models")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over a futuristic city, cyberpunk style")
    parser.add_argument("--output_path", type=str, default="../results/output.png")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    return parser.parse_args()

def main():
    args = parse_args()

    # Force CUDA to initialize and verify it's working before loading any model.
    # This catches GPU context issues early and ensures cuDNN can initialize properly.
    assert torch.cuda.is_available(), "CUDA not available!"
    device = torch.device("cuda")
    
    # Warm up CUDA/cuDNN with a dummy operation — this forces cuDNN to initialize
    # its internal handles before diffusers tries to use them during a conv forward pass.
    dummy = torch.zeros(1, device=device)
    del dummy
    torch.cuda.synchronize()
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading pipeline for {args.model_id}...")

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
     )
    
    # Enable aggressive CPU offload instead of moving the entire pipeline to GPU.
    # This prevents Out Of Memory errors when loading massive models like SD 3.5 on single GPUs
    # by moving individual components to the GPU only during the exact moment they run.
    pipe.enable_model_cpu_offload()

    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    pipe.enable_attention_slicing(1)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers memory efficient attention enabled.")
    except Exception as e:
        print(f"Could not enable xformers: {e}")

    if "turbo" in args.model_id.lower():
        print("Turbo model detected - adjusting inference steps and guidance scale.")
        args.num_inference_steps = min(args.num_inference_steps, 4)
        args.guidance_scale = 0.0

    print(f"Generating image with prompt: '{args.prompt}'")
    print(f"Steps: {args.num_inference_steps}, CFG: {args.guidance_scale}")

    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    ).images[0]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    image.save(args.output_path)
    print(f"Saved generated image to {args.output_path}")

if __name__ == "__main__":
    main()