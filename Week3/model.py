import torch
from torch import nn
from transformers import ResNetModel, CLIPVisionModel
import torchvision.models as tvm
from dataset import NUM_CHAR, char2idx, TEXT_MAX_LEN

try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig
    )
    HAS_XLSTM = True
except ImportError:
    HAS_XLSTM = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

GRU_DIM = 512
EMBED_DIM = 512
DECODER_LAYERS = 1
DECODER_TYPE = 'gru'

# Supported encoders: name -> (source, model_id, output_feature_dim)
ENCODER_CONFIGS = {
    'resnet18':        ('hf',          'microsoft/resnet-18',                    512),
    'resnet34':        ('hf',          'microsoft/resnet-34',                    512),
    'resnet50':        ('hf',          'microsoft/resnet-50',                   2048),
    'vgg16':           ('torchvision', 'vgg16',                                  512),
    'vgg19':           ('torchvision', 'vgg19',                                  512),
    'efficientnet_b0': ('torchvision', 'efficientnet_b0',                       1280),
    # CLIP encoders — features aligned with language, excellent for captioning.
    # Using openai/ variants with use_safetensors=True to bypass torch.load CVE-2025-32434.
    'clip-vit-b32':    ('clip',        'openai/clip-vit-base-patch32',           768),
    'clip-vit-l14':    ('clip',        'openai/clip-vit-large-patch14',         1024),
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

    def __init__(self, encoder_name='resnet18', freeze_encoder=False, 
                 decoder_type='gru', decoder_dim=512, decoder_layers=1, embed_dim=512):
        super().__init__()
        self.decoder_type = decoder_type
        self.decoder_dim = decoder_dim
        self.decoder_layers = decoder_layers
        self.embed_dim = embed_dim

        if encoder_name not in ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder '{encoder_name}'. Choose from: {list(ENCODER_CONFIGS)}")

        source, identifier, enc_dim = ENCODER_CONFIGS[encoder_name]

        if source == 'hf':
            self.encoder = ResNetModel.from_pretrained(identifier)
            self._is_hf = True
        elif source == 'clip':
            # use_safetensors=True bypasses torch.load (CVE-2025-32434 restriction on older PyTorch)
            self.encoder = CLIPVisionModel.from_pretrained(identifier, use_safetensors=True)
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

        self.encoder_proj = nn.Linear(enc_dim, decoder_dim) if enc_dim != decoder_dim else nn.Identity()
        
        if decoder_type == 'gru':
            self.decoder = nn.GRU(embed_dim, decoder_dim, num_layers=decoder_layers, batch_first=True)
        elif decoder_type == 'lstm':
            self.decoder = nn.LSTM(embed_dim, decoder_dim, num_layers=decoder_layers, batch_first=True)
        elif decoder_type == 'xlstm':
            if not HAS_XLSTM:
                raise ImportError("xLSTM implementation not found. Please install the 'xlstm' package.")
            
            # xLSTM (specifically mLSTM) configuration:
            # - We use a stack of mLSTM blocks (controlled by decoder_layers).
            # - embedding_dim is set to decoder_dim.
            # - context_length is set to TEXT_MAX_LEN + 1 (to account for the image prefix).
            xlstm_cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, 
                        num_heads=4
                    )
                ),
                num_blocks=decoder_layers,
                embedding_dim=decoder_dim,
                slstm_at=[], # pure mLSTM stack
                context_length=TEXT_MAX_LEN + 1,
                add_post_blocks_norm=True
            )
            self.decoder = xLSTMBlockStack(xlstm_cfg)
            
            # Since embed_dim and decoder_dim may differ, we ensure the input matches xLSTM's expected dim.
            self.decoder_proj_inp = nn.Linear(embed_dim, decoder_dim) if embed_dim != decoder_dim else nn.Identity()
        else:
            raise ValueError(f"Unknown decoder_type '{decoder_type}'. Choose from: 'gru', 'lstm', 'xlstm'")

        self.proj  = nn.Linear(decoder_dim, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, embed_dim)

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
        # Initialize hidden state with image features
        # Hidden state shape for GRU/LSTM: (num_layers, batch, hidden_dim)
        # We repeat the pooled image feature for each layer
        hidden = feat.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        
        if self.decoder_type == 'lstm':
            # LSTM also needs a cell state
            cell = torch.zeros_like(hidden)
            hidden = (hidden, cell)
        elif self.decoder_type == 'xlstm':
            # For xLSTM, we trigger the first "step" with the image feature as a virtual token
            # We initialize the state by passing the image feature through 'step'
            img_token = feat.unsqueeze(1) # (B, 1, decoder_dim)
            _, hidden = self.decoder.step(img_token, state=None)
            # After this, 'hidden' contains the compiled state from the image
        else:
            hidden = None
        if target_caption is not None:
            if self.decoder_type == 'xlstm':
                # For xLSTM, we prepend the image feature as the first "token" in the sequence
                # feat is (B, decoder_dim)
                img_token = feat.unsqueeze(1) # (B, 1, decoder_dim)
                embedded_seq = self.embed(target_caption[:, :-1]) # (B, T-1, embed_dim)
                embedded_seq = self.decoder_proj_inp(embedded_seq) # (B, T-1, decoder_dim)
                
                full_seq = torch.cat([img_token, embedded_seq], dim=1) # (B, T, decoder_dim)
                output = self.decoder(full_seq) # (B, T, decoder_dim)
                # Take all outputs corresponding to text tokens (index 1 onwards)
                output = output[:, 1:] # (B, T-1, decoder_dim)
                res = self.proj(output)
                return res.permute(0, 2, 1)
            else:
                # Teacher-forcing training: feed all ground-truth tokens in one decoder call
                inp_seq = self.embed(target_caption[:, :-1])  # (B, T-1, embed_dim)
                # nn.GRU/LSTM with batch_first=True expects (B, T, embed_dim)
                output, _ = self.decoder(inp_seq, hidden)     # (B, T-1, decoder_dim)
                res = self.proj(output)                       # (B, T-1, NUM_CHAR)
                return res.permute(0, 2, 1)                   # (B, NUM_CHAR, T-1)
        else:
            # Greedy auto-regressive inference with early EOS stopping
            curr_token = torch.full((batch_size,), char2idx['<SOS>'], device=device, dtype=torch.long)
            all_preds  = []
            eos_idx    = char2idx['<EOS>']
            finished   = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(TEXT_MAX_LEN - 1):
                inp = self.embed(curr_token).unsqueeze(1)  # (B, 1, embed_dim)
                
                if self.decoder_type == 'xlstm':
                    inp = self.decoder_proj_inp(inp)       # (B, 1, decoder_dim)
                    out, hidden = self.decoder.step(inp, state=hidden)
                else:
                    out, hidden = self.decoder(inp, hidden) # (B, 1, decoder_dim)

                logits = self.proj(out.squeeze(1))            # (B, NUM_CHAR)
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
    parser.add_argument('--decoder_type', default='gru', choices=['gru', 'lstm', 'xlstm'])
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=512)
    args = parser.parse_args()

    model = ImageCaptioningModel(
        encoder_name=args.encoder, 
        freeze_encoder=args.freeze,
        decoder_type=args.decoder_type,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.layers,
        embed_dim=args.embed_dim
    ).to(DEVICE)
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
    out = model(dummy)
    print(f"Encoder: {args.encoder} | Decoder: {args.decoder_type} | Output shape: {out.shape}")
    assert out.shape[0] == 2 and out.shape[1] == NUM_CHAR and out.shape[2] <= TEXT_MAX_LEN - 1
    print("Model forward pass successful.")
