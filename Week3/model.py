import torch
from torch import nn
from transformers import ResNetModel, CLIPVisionModel, CLIPTextModel
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
      - embed: token embedding (vocab_size, GRU_DIM)
      - proj: linear output head (GRU_DIM → vocab_size)

    Training uses teacher forcing; inference is greedy auto-regressive with early EOS stopping.
    """

    def __init__(self, encoder_name='resnet18', freeze_encoder=False, 
                 decoder_type='gru', decoder_dim=512, decoder_layers=1, embed_dim=512,
                 vocab_size=NUM_CHAR, sos_idx=char2idx['<SOS>'], eos_idx=char2idx['<EOS>'], 
                 pad_idx=char2idx['<PAD>'], max_len=TEXT_MAX_LEN,
                 clip_embeddings=False, clip_model_id='openai/clip-vit-base-patch32',
                 freeze_embeddings=False):
        super().__init__()
        self.decoder_type = decoder_type
        self.decoder_dim = decoder_dim
        self.decoder_layers = decoder_layers
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.max_len = max_len

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
                context_length=self.max_len + 1,
                add_post_blocks_norm=True
            )
            self.decoder = xLSTMBlockStack(xlstm_cfg)
            
            # Since embed_dim and decoder_dim may differ, we ensure the input matches xLSTM's expected dim.
            self.decoder_proj_inp = nn.Linear(embed_dim, decoder_dim) if embed_dim != decoder_dim else nn.Identity()
        else:
            raise ValueError(f"Unknown decoder_type '{decoder_type}'. Choose from: 'gru', 'lstm', 'xlstm'")

        self.proj  = nn.Linear(decoder_dim, vocab_size)
        
        if clip_embeddings:
            print(f"[Model] Initializing embeddings from {clip_model_id}...")
            clip_text = CLIPTextModel.from_pretrained(clip_model_id, use_safetensors=True)
            pretrained_weights = clip_text.get_input_embeddings().weight.data
            
            # Check if vocab_size matches
            if pretrained_weights.shape[0] != vocab_size:
                print(f"[Warning] CLIP vocab size ({pretrained_weights.shape[0]}) != tokenizer vocab size ({vocab_size}). Truncating/Padding.")
                # Basic handling: if tokenizer vocab is smaller, take first N tokens. 
                # CLIP tokenizer usually matches 49408 exactly.
                new_weights = torch.zeros((vocab_size, pretrained_weights.shape[1]))
                common = min(vocab_size, pretrained_weights.shape[0])
                new_weights[:common] = pretrained_weights[:common]
                pretrained_weights = new_weights

            self.embed = nn.Embedding.from_pretrained(pretrained_weights, freeze=freeze_embeddings)
            # Override embed_dim to match pretrained weights
            self.embed_dim = pretrained_weights.shape[1]
            print(f"[Model] Embed dim set to {self.embed_dim} from CLIP.")
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)

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
        hidden = feat.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        
        if self.decoder_type == 'lstm':
            cell = torch.zeros_like(hidden)
            hidden = (hidden, cell)
        elif self.decoder_type == 'xlstm':
            img_token = feat.unsqueeze(1) # (B, 1, decoder_dim)
            _, hidden = self.decoder.step(img_token, state=None)
        else:
            hidden = None

        if target_caption is not None:
            if self.decoder_type == 'xlstm':
                img_token = feat.unsqueeze(1) # (B, 1, decoder_dim)
                embedded_seq = self.embed(target_caption[:, :-1]) # (B, T-1, embed_dim)
                embedded_seq = self.decoder_proj_inp(embedded_seq) # (B, T-1, decoder_dim)
                
                full_seq = torch.cat([img_token, embedded_seq], dim=1) # (B, T, decoder_dim)
                output = self.decoder(full_seq) # (B, T, decoder_dim)
                output = output[:, 1:] # (B, T-1, decoder_dim)
                res = self.proj(output)
                return res.permute(0, 2, 1)
            else:
                inp_seq = self.embed(target_caption[:, :-1])  # (B, T-1, embed_dim)
                output, _ = self.decoder(inp_seq, hidden)     # (B, T-1, decoder_dim)
                res = self.proj(output)                       # (B, T-1, vocab_size)
                return res.permute(0, 2, 1)                   # (B, vocab_size, T-1)
        else:
            curr_token = torch.full((batch_size,), self.sos_idx, device=device, dtype=torch.long)
            all_preds  = []
            eos_idx    = self.eos_idx
            finished   = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(self.max_len - 1):
                inp = self.embed(curr_token).unsqueeze(1)  # (B, 1, embed_dim)
                
                if self.decoder_type == 'xlstm':
                    inp = self.decoder_proj_inp(inp)       # (B, 1, decoder_dim)
                    out, hidden = self.decoder.step(inp, state=hidden)
                else:
                    out, hidden = self.decoder(inp, hidden) # (B, 1, decoder_dim)

                logits = self.proj(out.squeeze(1))            # (B, vocab_size)
                all_preds.append(logits.unsqueeze(2))
                curr_token = logits.argmax(dim=1)
                finished = finished | (curr_token == eos_idx)
                if finished.all():
                    break

            return torch.cat(all_preds, dim=2)  # (B, vocab_size, ≤max_len-1)



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
