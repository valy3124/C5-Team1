import torch
from torch import nn
from transformers import ResNetModel
import torchvision.models as tvm
from dataset import NUM_CHAR, char2idx, TEXT_MAX_LEN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

GRU_DIM = 512

# Supported encoders: name -> (source, identifier, output feature dim)
ENCODER_CONFIGS = {
    'resnet18': ('hf',          'microsoft/resnet-18', 512),
    'resnet34': ('hf',          'microsoft/resnet-34', 512),
    'resnet50': ('hf',          'microsoft/resnet-50', 2048),
    'vgg16':    ('torchvision', 'vgg16',               512),
    'vgg19':    ('torchvision', 'vgg19',               512),
    'efficientnet_b0': ('torchvision', 'efficientnet_b0', 1280),
}


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder_name='resnet18'):
        super().__init__()
        if encoder_name not in ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder '{encoder_name}'. Choose from: {list(ENCODER_CONFIGS)}")

        source, identifier, enc_dim = ENCODER_CONFIGS[encoder_name]

        if source == 'hf':
            self.encoder = ResNetModel.from_pretrained(identifier)
            self._is_hf = True
        else:
            # Use convolutional features + global average pool → (B, enc_dim, 1, 1)
            net = getattr(tvm, identifier)(weights='DEFAULT')
            self.encoder = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
            self._is_hf = False

        # Project encoder output to GRU hidden dim if needed
        self.encoder_proj = nn.Linear(enc_dim, GRU_DIM) if enc_dim != GRU_DIM else nn.Identity()

        # Decoder: 1-layer GRU
        self.gru = nn.GRU(GRU_DIM, GRU_DIM, num_layers=1)
        self.proj = nn.Linear(GRU_DIM, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, GRU_DIM)

    def _extract_features(self, img):
        if self._is_hf:
            return self.encoder(img).pooler_output.flatten(1)  # (B, enc_dim)
        return self.encoder(img).flatten(1)                    # (B, 512)

    def forward(self, img, target_caption=None):
        batch_size = img.shape[0]
        device = img.device

        # Encode image → (1, B, GRU_DIM)
        feat = self._extract_features(img)
        feat = self.encoder_proj(feat)
        hidden = feat.unsqueeze(0)  # (1, B, GRU_DIM)

        if target_caption is not None:
            # --- Training Mode (Teacher Forcing): O(N) ---
            # Input: <SOS> token and all caption characters except the last one
            inp_seq = self.embed(target_caption[:, :-1]) # (B, TEXT_MAX_LEN-1, GRU_DIM)
            inp_seq = inp_seq.permute(1, 0, 2)           # (TEXT_MAX_LEN-1, B, GRU_DIM)
            
            output, _ = self.gru(inp_seq, hidden)        # (TEXT_MAX_LEN-1, B, GRU_DIM)
            res = self.proj(output.permute(1, 0, 2))      # (B, TEXT_MAX_LEN-1, NUM_CHAR)
            return res.permute(0, 2, 1)                  # (B, NUM_CHAR, TEXT_MAX_LEN-1)
        else:
            # --- Inference Mode (Auto-regressive): O(N) ---
            curr_token = torch.full((batch_size,), char2idx['<SOS>'], device=device, dtype=torch.long)
            all_preds = []

            for _ in range(TEXT_MAX_LEN - 1):
                inp = self.embed(curr_token).unsqueeze(0) # (1, B, GRU_DIM)
                out, hidden = self.gru(inp, hidden)       # (1, B, GRU_DIM)
                logits = self.proj(out.squeeze(0))        # (B, NUM_CHAR)
                all_preds.append(logits.unsqueeze(2))     # (B, NUM_CHAR, 1)
                curr_token = logits.argmax(dim=1)         # Greedy selection
            
            return torch.cat(all_preds, dim=2)            # (B, NUM_CHAR, TEXT_MAX_LEN-1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='resnet18', choices=list(ENCODER_CONFIGS))
    args = parser.parse_args()

    model = ImageCaptioningModel(encoder_name=args.encoder).to(DEVICE)
    dummy_img = torch.randn(2, 3, 224, 224).to(DEVICE)
    out = model(dummy_img)
    print(f"Encoder: {args.encoder} | Output shape: {out.shape}")
    assert out.shape == (2, NUM_CHAR, TEXT_MAX_LEN)
    print("Model forward pass successful.")
