import torch
from model import ImageCaptioningModel

def test_beam_search():
    print("Setting up model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the model with LSTM and Adaptive Attention (small version for fast test)
    model = ImageCaptioningModel(
        encoder_name='resnet18',
        decoder_type='lstm',
        decoder_dim=128,
        decoder_layers=1,
        embed_dim=128,
        attn_type='adaptive',
        attn_dim=64,
        vocab_size=100,  # small vocab
        max_len=10
    ).to(device)
    
    model.eval()
    
    print("Running forward pass with greedy...")
    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        out_greedy = model(dummy_img, generation_method='greedy')
        print("Greedy Output shape:", out_greedy.shape)
        
        print("Running forward pass with beam search...")
        out_beam = model(dummy_img, generation_method='beam', beam_size=3)
        print("Beam Output shape:", out_beam.shape)
        
    print("SUCCESS!")

if __name__ == "__main__":
    test_beam_search()
