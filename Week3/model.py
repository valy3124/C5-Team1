import torch
from torch import nn
from transformers import ResNetModel
from dataset import NUM_CHAR, char2idx, TEXT_MAX_LEN

# Move DEVICE to a configuration or rely on passed argument if necessary
# For simplicity in this structure we define it here, though it's typically passed from train.py
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageCaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: Baseline is ResNet-18
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18')
        
        # Decoder: Baseline is 1-layer GRU
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)

    def forward(self, img):
        batch_size = img.shape[0]
        
        # Extract features from image
        feat = self.resnet(img)
        # feat.pooler_output is (batch, 512, 1, 1).
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # shape: (1, batch_size, 512)
        
        # Initialize decoder step with <SOS>
        # Move device assignment dynamically based on where img is
        device = img.device
        start = torch.tensor(char2idx['<SOS>'], device=device)
        start_embed = self.embed(start) # shape: (512,)
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # shape: (1, batch_size, 512)
        
        inp = start_embeds
        hidden = feat
        
        # Generative loop mimicking baseline behavior (without teacher forcing natively in this forward)
        for t in range(TEXT_MAX_LEN - 1): # Remove <SOS> from max len
            out, hidden = self.gru(inp, hidden)
            # Take the very last output step and concatenate it to input
            inp = torch.cat((inp, out[-1:]), dim=0) # shape: (t+2, batch_size, 512)
            
        res = inp.permute(1, 0, 2) # shape: (batch_size, seq_len, 512)
        res = self.proj(res) # shape: (batch_size, seq_len, NUM_CHAR)
        res = res.permute(0, 2, 1) # shape: (batch_size, NUM_CHAR, seq_len)
        
        return res

if __name__ == "__main__":
    model = ImageCaptioningModel().to(DEVICE)
    dummy_img = torch.randn(2, 3, 224, 224).to(DEVICE)
    out = model(dummy_img)
    print("Output shape:", out.shape)
    assert out.shape == (2, NUM_CHAR, TEXT_MAX_LEN)
    print("Model forward pass successful.")
