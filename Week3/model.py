import torch
from torch import nn
from transformers import ResNetModel
import torchvision.models as tvm
from dataset import NUM_CHAR, char2idx, TEXT_MAX_LEN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

GRU_DIM = 512

# Supported encoders: name -> (source, model_id, output_feature_dim)
ENCODER_CONFIGS = {
    'resnet18':        ('hf',          'microsoft/resnet-18',    512),
    'resnet34':        ('hf',          'microsoft/resnet-34',    512),
    'resnet50':        ('hf',          'microsoft/resnet-50',   2048),
    'vgg16':           ('torchvision', 'vgg16',                  512),
    'vgg19':           ('torchvision', 'vgg19',                  512),
    'efficientnet_b0': ('torchvision', 'efficientnet_b0',       1280),
}


class ImageCaptioningModel(nn.Module):
    """
    Encoder-decoder image captioning model.

    Architecture:
      - Encoder: pre-trained CNN (ResNet / VGG / EfficientNet) → pooled feature vector (B, enc_dim)
      - encoder_proj: linear projection to GRU_DIM if enc_dim != GRU_DIM, else Identity
      - Decoder: single-layer GRU initialised with the image feature as h0
      - embed: character embedding (NUM_CHAR, GRU_DIM)
      - proj: linear output head (GRU_DIM → NUM_CHAR)

    Training uses teacher forcing; inference is greedy auto-regressive with early EOS stopping.
    """

    def __init__(self, encoder_name='resnet18', freeze_encoder=False):
        super().__init__()
        if encoder_name not in ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder '{encoder_name}'. Choose from: {list(ENCODER_CONFIGS)}")

        source, identifier, enc_dim = ENCODER_CONFIGS[encoder_name]

        if source == 'hf':
            self.encoder = ResNetModel.from_pretrained(identifier)
            self._is_hf = True
        else:
            net = getattr(tvm, identifier)(weights='DEFAULT')
            self.encoder = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
            self._is_hf = False

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"[Model] Encoder '{encoder_name}' is FROZEN.")
        else:
            print(f"[Model] Encoder '{encoder_name}' is TRAINABLE.")

        self.encoder_proj = nn.Linear(enc_dim, GRU_DIM) if enc_dim != GRU_DIM else nn.Identity()
        self.gru   = nn.GRU(GRU_DIM, GRU_DIM, num_layers=1)
        self.proj  = nn.Linear(GRU_DIM, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, GRU_DIM)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[Model] Trainable params: {trainable:,} / {total:,}")

    def _extract_features(self, img):
        if self._is_hf:
            return self.encoder(img).pooler_output.flatten(1)  # (B, enc_dim)
        return self.encoder(img).flatten(1)                    # (B, enc_dim)

    def forward(self, img, target_caption=None):
        batch_size = img.shape[0]
        device = img.device

        feat = self._extract_features(img)
        feat = self.encoder_proj(feat)
        hidden = feat.unsqueeze(0)  # (1, B, GRU_DIM) — initial GRU hidden state

        if target_caption is not None:
            # Teacher-forcing training: feed all ground-truth tokens in one GRU call
            inp_seq = self.embed(target_caption[:, :-1])  # (B, T-1, GRU_DIM)
            inp_seq = inp_seq.permute(1, 0, 2)            # (T-1, B, GRU_DIM)
            output, _ = self.gru(inp_seq, hidden)         # (T-1, B, GRU_DIM)
            res = self.proj(output.permute(1, 0, 2))      # (B, T-1, NUM_CHAR)
            return res.permute(0, 2, 1)                   # (B, NUM_CHAR, T-1)
        else:
            # Greedy auto-regressive inference with early EOS stopping
            curr_token = torch.full((batch_size,), char2idx['<SOS>'], device=device, dtype=torch.long)
            all_preds  = []
            eos_idx    = char2idx['<EOS>']
            finished   = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(TEXT_MAX_LEN - 1):
                inp    = self.embed(curr_token).unsqueeze(0)  # (1, B, GRU_DIM)
                out, hidden = self.gru(inp, hidden)           # (1, B, GRU_DIM)
                logits = self.proj(out.squeeze(0))            # (B, NUM_CHAR)
                all_preds.append(logits.unsqueeze(2))
                curr_token = logits.argmax(dim=1)
                finished = finished | (curr_token == eos_idx)
                if finished.all():
                    break

            return torch.cat(all_preds, dim=2)  # (B, NUM_CHAR, ≤TEXT_MAX_LEN-1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='resnet18', choices=list(ENCODER_CONFIGS))
    parser.add_argument('--freeze', action='store_true', help='Freeze encoder weights')
    args = parser.parse_args()

    model = ImageCaptioningModel(encoder_name=args.encoder, freeze_encoder=args.freeze).to(DEVICE)
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
    out = model(dummy)
    print(f"Encoder: {args.encoder} | Output shape: {out.shape}")
    assert out.shape[0] == 2 and out.shape[1] == NUM_CHAR and out.shape[2] <= TEXT_MAX_LEN - 1
    print("Model forward pass successful.")
