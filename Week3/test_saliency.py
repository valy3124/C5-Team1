import os
import sys
# Make sure we're in the right place
sys.path.append(os.path.dirname(__file__))

import torch
from model import ImageCaptioningModel
from saliency_plotter import plot_saliency_maps

def test():
    print("Testing saliency...")
    # Initialize basic model to test the PyTorch graph logic
    try:
        model = ImageCaptioningModel(
            encoder_name='resnet18',
            decoder_type='gru',
            attn_type='early_fusion',
            decoder_dim=128,
            embed_dim=128
        )
        print("Model initialized.")

        # Dummy image
        img = torch.randn(1, 3, 224, 224)
        print("Running generate_with_saliency...")
        result, saliency_maps = model.generate_with_saliency(img)
        
        print(f"Generated {len(saliency_maps)} saliency maps.")
        if len(saliency_maps) > 0:
            print(f"Shape of first map: {saliency_maps[0].shape}")
        
        # Get generated tokens
        tokens = result[0].argmax(dim=0).tolist()
        
        print("Testing plotting...")
        save_path = os.path.join(os.path.dirname(__file__), "test_sloop.png")
        plot_saliency_maps(img, saliency_maps, tokens, save_path)
        print(f"Plot saved to {save_path}")
        print("All tests passed.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test()
