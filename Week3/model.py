import math
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


# ---------------------------------------------------------------------------
# Bahdanau (additive) attention
# Reference: "Show, Attend and Tell" (Xu et al., 2015)
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    """
    Additive attention over spatial encoder features.

    At each decoder step t:
        e_i   = score_fc( tanh( enc_proj(a_i) + dec_proj(h_{t-1}) ) )   for each region i
        alpha  = softmax(e)                                               # (B, L)
        context = sum_i( alpha_i * a_i )                                 # (B, enc_dim)

    Both features and hidden state are expected to be projected to `decoder_dim`
    before calling this module (encoder_proj handles this upstream).
    """

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, attn_dim, bias=False)
        self.dec_proj = nn.Linear(dec_dim, attn_dim, bias=False)
        self.score_fc = nn.Linear(attn_dim, 1,       bias=False)

    def forward(self, features: torch.Tensor, h: torch.Tensor):
        """
        Args:
            features : (B, L, enc_dim)  – spatial encoder features (post encoder_proj)
            h        : (B, dec_dim)     – decoder hidden state at previous step
        Returns:
            context  : (B, enc_dim)     – attention-weighted feature vector
            alpha    : (B, L)           – attention weights (sum to 1)
        """
        # enc_proj broadcasts over L; dec_proj unsqueezed to broadcast over L
        energy = self.score_fc(
            torch.tanh(
                self.enc_proj(features) +          # (B, L, attn_dim)
                self.dec_proj(h).unsqueeze(1)      # (B, 1, attn_dim)
            )
        ).squeeze(2)                               # (B, L)

        alpha   = torch.softmax(energy, dim=1)                    # (B, L)
        context = (alpha.unsqueeze(2) * features).sum(dim=1)      # (B, enc_dim)
        return context, alpha


# ---------------------------------------------------------------------------
# Adaptive Attention with Visual Sentinel
# Reference: "Knowing When to Look" (Lu et al., 2017)
# ---------------------------------------------------------------------------

class AdaptiveAttention(nn.Module):
    """
    Adaptive Attention using a visually-derived sentinel.
    Computes a sentinel vector s_t from the hidden state.
    Then computes attention over features and s_t.
    """
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, attn_dim, bias=False)
        self.dec_proj = nn.Linear(dec_dim, attn_dim, bias=False)
        self.sentinel_proj = nn.Linear(dec_dim, enc_dim, bias=False)  # Map hidden to visual space
        
        # We share the attention projection for features and sentinel
        self.score_fc = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, features: torch.Tensor, h: torch.Tensor):
        """
        Args:
            features : (B, L, enc_dim)
            h        : (B, dec_dim)
        Returns:
            context  : (B, enc_dim)
            alpha    : (B, L+1) attention weights (last element is sentinel beta_t)
        """
        B, L, _ = features.shape
        
        # 1. Compute sentinel s_t from hidden state
        # A simple approximation of the sentinel gate: project h_t to visual space and apply tanh
        s_t = torch.tanh(self.sentinel_proj(h)).unsqueeze(1)  # (B, 1, enc_dim)
        
        # 2. Concatenate features and sentinel
        extended_features = torch.cat([features, s_t], dim=1) # (B, L+1, enc_dim)
        
        # 3. Compute attention scores over extended features
        energy = self.score_fc(
            torch.tanh(
                self.enc_proj(extended_features) +  # (B, L+1, attn_dim)
                self.dec_proj(h).unsqueeze(1)       # (B, 1, attn_dim)
            )
        ).squeeze(2)                                # (B, L+1)
        
        alpha = torch.softmax(energy, dim=1)        # (B, L+1)
        
        # 4. Context vector is weighted sum
        context = (alpha.unsqueeze(2) * extended_features).sum(dim=1) # (B, enc_dim)
        
        return context, alpha


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ImageCaptioningModel(nn.Module):
    """
    Encoder-decoder image captioning model.

    Without attention:
      - Encoder → pooled feature vector (B, enc_dim)
      - encoder_proj → (B, decoder_dim)
      - GRU/LSTM/xLSTM decoder, input = embed(token)

    With attention (use_attention=True):
      - Encoder → spatial feature map (B, L, enc_dim)
      - encoder_proj → (B, L, decoder_dim)
      - h0 = mean over spatial locations
      - At each step: context = attention(features, h_prev)
      - Decoder input = cat([embed(token), context], dim=-1)
      - Grid size L = H*W (e.g. 7×7=49 for most CNN encoders)

    Note: xLSTM decoder is NOT compatible with attention mode.

    Training uses teacher forcing; inference is greedy auto-regressive.
    """

    def __init__(
        self,
        encoder_name='resnet18',
        freeze_encoder=False,
        decoder_type='gru',
        decoder_dim=512,
        decoder_layers=1,
        embed_dim=512,
        vocab_size=NUM_CHAR,
        sos_idx=char2idx['<SOS>'],
        eos_idx=char2idx['<EOS>'],
        pad_idx=char2idx['<PAD>'],
        max_len=TEXT_MAX_LEN,
        clip_embeddings=False,
        clip_model_id='openai/clip-vit-base-patch32',
        freeze_embeddings=False,
        attn_type=None,
        attn_dim=256,
    ):
        super().__init__()

        # ---------- guards ----------
        if attn_type not in [None, 'soft', 'adaptive', 'early_fusion']:
            raise ValueError(f"Unknown attn_type '{attn_type}'. Choose from: None, 'soft', 'adaptive', 'early_fusion'")
        if encoder_name not in ENCODER_CONFIGS:
            raise ValueError(
                f"Unknown encoder '{encoder_name}'. "
                f"Choose from: {list(ENCODER_CONFIGS)}"
            )

        # ---------- store config ----------
        self.decoder_type   = decoder_type
        self.decoder_dim    = decoder_dim
        self.decoder_layers = decoder_layers
        self.embed_dim      = embed_dim
        self.vocab_size     = vocab_size
        self.sos_idx        = sos_idx
        self.eos_idx        = eos_idx
        self.pad_idx        = pad_idx
        self.max_len        = max_len
        self.attn_type      = attn_type
        self.use_attention  = attn_type is not None

        if self.attn_type == 'early_fusion':
            self.visual_separator = nn.Parameter(torch.randn(1, 1, decoder_dim))

        # ---------- encoder ----------
        source, identifier, enc_dim = ENCODER_CONFIGS[encoder_name]
        self._clip_encoder = (source == 'clip')

        if source == 'hf':
            self.encoder = ResNetModel.from_pretrained(identifier)
            self._is_hf  = True
        elif source == 'clip':
            self.encoder = CLIPVisionModel.from_pretrained(
                identifier, use_safetensors=True
            )
            self._is_hf = True
        else:  # torchvision
            net = getattr(tvm, identifier)(weights='DEFAULT')
            if self.use_attention:
                # Preserve spatial map — strip the adaptive avg-pool
                self.encoder = net.features
            else:
                self.encoder = nn.Sequential(
                    net.features, nn.AdaptiveAvgPool2d(1)
                )
            self._is_hf = False

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print(f"[Model] Encoder '{encoder_name}' is FROZEN.")
        else:
            print(f"[Model] Encoder '{encoder_name}' is TRAINABLE.")

        # ---------- encoder projection ----------
        # Works for both (B, enc_dim) and (B, L, enc_dim) — Linear acts on last dim
        self.encoder_proj = (
            nn.Linear(enc_dim, decoder_dim) if enc_dim != decoder_dim
            else nn.Identity()
        )

        # ---------- attention ----------
        if self.attn_type == 'soft':
            self.attention = BahdanauAttention(
                enc_dim=decoder_dim,
                dec_dim=decoder_dim,
                attn_dim=attn_dim,
            )
            decoder_input_size = embed_dim + decoder_dim   # embed ‖ context
            
        elif self.attn_type == 'adaptive':
            self.attention = AdaptiveAttention(
                enc_dim=decoder_dim,
                dec_dim=decoder_dim,
                attn_dim=attn_dim,
            )
            decoder_input_size = embed_dim                 # input is just embed, context applied later
            # For adaptive attention, we mix context and hidden state to feed into final projection
            self.adaptive_context_proj = nn.Linear(decoder_dim + decoder_dim, decoder_dim)
            
        else:
            self.attention = None
            decoder_input_size = embed_dim

        # ---------- decoder ----------
        if decoder_type == 'gru':
            self.decoder = nn.GRU(
                decoder_input_size, decoder_dim,
                num_layers=decoder_layers, batch_first=True
            )
        elif decoder_type == 'lstm':
            self.decoder = nn.LSTM(
                decoder_input_size, decoder_dim,
                num_layers=decoder_layers, batch_first=True
            )
        elif decoder_type == 'xlstm':
            if not HAS_XLSTM:
                raise ImportError(
                    "xLSTM not found. Install the 'xlstm' package."
                )
            xlstm_cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, num_heads=4)
                ),
                num_blocks=decoder_layers,
                embedding_dim=decoder_dim,
                slstm_at=[],
                context_length=self.max_len + 512,
                add_post_blocks_norm=True,
            )
            self.decoder = xLSTMBlockStack(xlstm_cfg)
            self.decoder_proj_inp = (
                nn.Linear(embed_dim, decoder_dim)
                if embed_dim != decoder_dim else nn.Identity()
            )
        else:
            raise ValueError(
                f"Unknown decoder_type '{decoder_type}'. "
                f"Choose from: 'gru', 'lstm', 'xlstm'"
            )

        # ---------- output projection ----------
        self.proj = nn.Linear(decoder_dim, vocab_size)

        # ---------- token embeddings ----------
        if clip_embeddings:
            print(f"[Model] Initializing embeddings from {clip_model_id}...")
            clip_text = CLIPTextModel.from_pretrained(
                clip_model_id, use_safetensors=True
            )
            pretrained_weights = clip_text.get_input_embeddings().weight.data
            if pretrained_weights.shape[0] != vocab_size:
                print(
                    f"[Warning] CLIP vocab size ({pretrained_weights.shape[0]}) "
                    f"!= tokenizer vocab size ({vocab_size}). Truncating/Padding."
                )
                new_weights = torch.zeros((vocab_size, pretrained_weights.shape[1]))
                common = min(vocab_size, pretrained_weights.shape[0])
                new_weights[:common] = pretrained_weights[:common]
                pretrained_weights = new_weights
            self.embed = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=freeze_embeddings
            )
            self.embed_dim = pretrained_weights.shape[1]
            print(f"[Model] Embed dim set to {self.embed_dim} from CLIP.")
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        attn_tag  = " [+Attention]" if self.use_attention else ""
        print(f"[Model{attn_tag}] Trainable params: {trainable:,} / {total:,}")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            use_attention=False  →  (B, enc_dim)      pooled global feature
            use_attention=True   →  (B, L, enc_dim)   spatial feature map
        """
        if self._clip_encoder:
            # last_hidden_state: (B, N+1, D)  — index 0 is CLS token
            lhs = self.encoder(img).last_hidden_state
            if self.use_attention:
                return lhs[:, 1:, :]   # drop CLS → (B, N, D)  e.g. (B,49,768)
            else:
                return lhs[:, 0, :]    # CLS as global repr → (B, D)

        elif self._is_hf:  # ResNet via HuggingFace
            out = self.encoder(img)
            if self.use_attention:
                # last_hidden_state: (B, C, H, W)
                fmap = out.last_hidden_state          # (B, C, H, W)
                B, C, H, W = fmap.shape
                return fmap.permute(0, 2, 3, 1).reshape(B, H * W, C)
            else:
                return out.pooler_output.flatten(1)   # (B, enc_dim)

        else:  # torchvision (features only, with or without pool)
            out = self.encoder(img)                   # (B, C, H, W) or (B, C, 1, 1)
            if self.use_attention:
                B, C, H, W = out.shape
                return out.permute(0, 2, 3, 1).reshape(B, H * W, C)
            else:
                return out.flatten(1)                 # (B, enc_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        img: torch.Tensor,
        target_caption: torch.Tensor = None,
        return_attention: bool = False,
        generation_method: str = "greedy",
        beam_size: int = 3,
    ):
        """
        Args:
            img             : (B, 3, H, W)
            target_caption  : (B, T) for teacher-forcing; None → greedy inference
            return_attention: if True and use_attention, returns (logits, list_of_alphas)
                              Each alpha in the list has shape (B, L).
        Returns (teacher-forcing):
            logits : (B, vocab_size, T-1)
        Returns (greedy, no attention):
            logits : (B, vocab_size, ≤max_len-1)
        Returns (greedy, with attention, return_attention=True):
            (logits, alphas)  where alphas = list[(B, L)] one per generated step
        """
        batch_size = img.shape[0]
        device     = img.device

        # ---- Encode ----
        features = self._extract_features(img)     # (B, enc_dim) or (B, L, enc_dim)
        features = self.encoder_proj(features)     # project to decoder_dim

        # ---- Initial decoder hidden state ----
        if self.use_attention and self.attn_type != 'early_fusion':
            h0 = features.mean(dim=1)              # (B, dec_dim) — mean over spatial
        elif self.attn_type == 'early_fusion':
            h0 = None
        else:
            h0 = features                          # (B, dec_dim)

        if h0 is not None:
            hidden = h0.unsqueeze(0).repeat(self.decoder_layers, 1, 1)  # (layers, B, dec_dim)
            if self.decoder_type == 'lstm':
                cell   = torch.zeros_like(hidden)
                hidden = (hidden, cell)
            elif self.decoder_type == 'xlstm':
                img_token = h0.unsqueeze(1)            # (B, 1, dec_dim)
                _, hidden = self.decoder.step(img_token, state=None)
        else:
            hidden = None

        # ==============================================================
        #  TEACHER FORCING  (training)
        # ==============================================================
        if target_caption is not None:

            if self.decoder_type == 'xlstm' and not self.use_attention:
                # xLSTM path basic (no attention)
                embedded_seq = self.embed(target_caption[:, :-1])
                embedded_seq = self.decoder_proj_inp(embedded_seq)
                full_seq     = torch.cat([h0.unsqueeze(1), embedded_seq], dim=1)
                output       = self.decoder(full_seq)
                output       = output[:, 1:]
                res          = self.proj(output)
                return res.permute(0, 2, 1)

            if self.attn_type == 'soft':
                # Soft attention: context is computed from h_{t-1} and concatenated to input
                embedded = self.embed(target_caption[:, :-1])  # (B, T-1, embed_dim)
                outputs  = []
                
                # If using xlstm, state is the hidden tuple
                for t in range(embedded.size(1)):
                    if self.decoder_type == 'xlstm':
                        # xLSTM has state as hidden tuple, we can use output as h_q
                        h_q = hidden[0].squeeze(1) if type(hidden) in [tuple, list] and len(hidden) > 0 and type(hidden[0]) == torch.Tensor else h0
                    else:
                        h_q = hidden[0][-1] if self.decoder_type == 'lstm' else hidden[-1]
                    
                    context, _ = self.attention(features, h_q)              # (B, dec_dim)
                    
                    if self.decoder_type == 'xlstm':
                        inp_t = torch.cat([embedded[:, t, :], context], dim=1).unsqueeze(1)
                        inp_t = self.decoder_proj_inp(inp_t)
                        out, hidden = self.decoder.step(inp_t, state=hidden)
                    else:
                        inp_t = torch.cat([embedded[:, t, :], context], dim=1).unsqueeze(1)
                        out, hidden = self.decoder(inp_t, hidden)                # (B,1,dec_dim)
                        
                    outputs.append(out)
                output = torch.cat(outputs, dim=1)                           # (B,T-1,dec_dim)
                
            elif self.attn_type == 'adaptive':
                # Adaptive attention: input goes into RNN -> h_t -> attention -> h_att
                embedded = self.embed(target_caption[:, :-1])
                if self.decoder_type == 'xlstm':
                    embedded = self.decoder_proj_inp(embedded)
                
                outputs = []
                for t in range(embedded.size(1)):
                    inp_t = embedded[:, t, :].unsqueeze(1)
                    if self.decoder_type == 'xlstm':
                        out, hidden = self.decoder.step(inp_t, state=hidden)
                        h_t = out.squeeze(1)
                    else:
                        out, hidden = self.decoder(inp_t, hidden)
                        h_t = hidden[0][-1] if self.decoder_type == 'lstm' else hidden[-1]
                        
                    context, _ = self.attention(features, h_t)
                    
                    h_att = torch.tanh(self.adaptive_context_proj(torch.cat([context, h_t], dim=-1)))
                    outputs.append(h_att.unsqueeze(1))
                    
                output = torch.cat(outputs, dim=1)
                
            elif self.attn_type == 'early_fusion':
                embedded = self.embed(target_caption[:, :-1])
                if self.decoder_type == 'xlstm':
                    embedded = self.decoder_proj_inp(embedded)
                
                sep = self.visual_separator.expand(batch_size, -1, -1)
                full_seq = torch.cat([features, sep, embedded], dim=1) # (B, L+1+T-1, dec_dim)
                if self.decoder_type == 'xlstm':
                    output = self.decoder(full_seq)
                    output = output[:, features.size(1) + 1:] # Only keep text preds
                else:
                    output, _ = self.decoder(full_seq, hidden)
                    output = output[:, features.size(1) + 1:]
                    
            else:
                inp_seq = self.embed(target_caption[:, :-1])                 # (B,T-1,embed)
                output, _ = self.decoder(inp_seq, hidden)                    # (B,T-1,dec_dim)

            res = self.proj(output)                                          # (B,T-1,vocab)
            return res.permute(0, 2, 1)                                      # (B,vocab,T-1)

        # ==============================================================
        #  GREEDY / BEAM INFERENCE
        # ==============================================================
        if target_caption is None and generation_method == "beam" and self.attn_type == "adaptive" and self.decoder_type in ['lstm', 'gru']:
            return self._beam_search_adaptive(img, features, h0, beam_size, return_attention)

        if self.attn_type == 'early_fusion':
            sep = self.visual_separator.expand(batch_size, -1, -1)
            full_prefix = torch.cat([features, sep], dim=1)
            for f_i in range(full_prefix.size(1)):
                inp_f = full_prefix[:, f_i:f_i+1, :]
                if self.decoder_type == 'xlstm':
                    _, hidden = self.decoder.step(inp_f, state=hidden)
                else:
                    _, hidden = self.decoder(inp_f, hidden)

        curr_token = torch.full(
            (batch_size,), self.sos_idx, device=device, dtype=torch.long
        )
        all_preds  = []
        all_alphas = [] if (return_attention and self.use_attention) else None
        finished   = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.max_len - 1):
            inp = self.embed(curr_token).unsqueeze(1)   # (B, 1, embed_dim)

            if self.attn_type == 'soft':
                if self.decoder_type == 'xlstm':
                    h_q = hidden[0].squeeze(1) if type(hidden) in [tuple, list] and len(hidden) > 0 and type(hidden[0]) == torch.Tensor else h0
                else:
                    h_q = hidden[0][-1] if self.decoder_type == 'lstm' else hidden[-1]
                    
                context, alpha = self.attention(features, h_q)             # (B,dec),(B,L)
                if all_alphas is not None:
                    all_alphas.append(alpha.detach().cpu())
                    
                inp_t = torch.cat([inp.squeeze(1), context], dim=1).unsqueeze(1)
                
                if self.decoder_type == 'xlstm':
                    inp_t = self.decoder_proj_inp(inp_t)
                    out, hidden = self.decoder.step(inp_t, state=hidden)
                else:
                    out, hidden = self.decoder(inp_t, hidden)
                    
            elif self.attn_type == 'adaptive':
                if self.decoder_type == 'xlstm':
                    inp_t = self.decoder_proj_inp(inp)
                    out, hidden = self.decoder.step(inp_t, state=hidden)
                    h_t = out.squeeze(1)
                else:
                    out, hidden = self.decoder(inp, hidden)
                    h_t = hidden[0][-1] if self.decoder_type == 'lstm' else hidden[-1]
                    
                context, alpha = self.attention(features, h_t)
                if all_alphas is not None:
                    all_alphas.append(alpha.detach().cpu())
                    
                h_att = torch.tanh(self.adaptive_context_proj(torch.cat([context, h_t], dim=-1)))
                out = h_att.unsqueeze(1)
            
            else:
                if self.decoder_type == 'xlstm':
                    inp = self.decoder_proj_inp(inp)
                    out, hidden = self.decoder.step(inp, state=hidden)
                else:
                    out, hidden = self.decoder(inp, hidden)

            logits     = self.proj(out.squeeze(1))                          # (B, vocab)
            all_preds.append(logits.unsqueeze(2))
            curr_token = logits.argmax(dim=1)
            finished   = finished | (curr_token == self.eos_idx)
            if finished.all():
                break

        result = torch.cat(all_preds, dim=2)                                # (B, vocab, T)
        if return_attention and self.use_attention:
            return result, all_alphas
        return result

    def _beam_search_adaptive(self, img, features, h0, beam_size, return_attention):
        device = img.device
        batch_size = img.shape[0]
        
        all_res = []
        for b in range(batch_size):
            feat_b = features[b:b+1]
            
            if h0 is not None:
                h0_b = h0[b:b+1]
                hidden = h0_b.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
                if self.decoder_type == 'lstm':
                    cell = torch.zeros_like(hidden)
                    hidden = (hidden, cell)
            else:
                hidden = None
                
            beams = [(0.0, [self.sos_idx], hidden)]
            
            for step in range(self.max_len - 1):
                new_beams = []
                for score, seq, h_state in beams:
                    if seq[-1] == self.eos_idx:
                        new_beams.append((score, seq, h_state))
                        continue
                        
                    curr_token_tensor = torch.tensor([seq[-1]], device=device, dtype=torch.long)
                    inp = self.embed(curr_token_tensor).unsqueeze(1)
                    
                    if self.decoder_type == 'lstm':
                        out, next_h_state = self.decoder(inp, h_state)
                        h_t = next_h_state[0][-1]
                    else:
                        out, next_h_state = self.decoder(inp, h_state)
                        h_t = next_h_state[-1]
                        
                    context, alpha = self.attention(feat_b, h_t)
                    
                    h_att = torch.tanh(self.adaptive_context_proj(torch.cat([context, h_t], dim=-1)))
                    out = h_att.unsqueeze(1)
                    
                    logits = self.proj(out.squeeze(1))
                    log_probs = torch.log_softmax(logits, dim=-1)[0]
                    
                    topk_probs, topk_idx = torch.topk(log_probs, beam_size)
                    
                    for i in range(beam_size):
                        next_score = score + topk_probs[i].item()
                        next_seq = seq + [topk_idx[i].item()]
                        new_beams.append((next_score, next_seq, next_h_state))
                        
                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
                if all(b[1][-1] == self.eos_idx for b in beams):
                    break
                    
            best_beam = beams[0]
            best_seq = best_beam[1][1:]
            
            if len(best_seq) < self.max_len - 1:
                best_seq.extend([self.pad_idx] * (self.max_len - 1 - len(best_seq)))
            else:
                best_seq = best_seq[:self.max_len - 1]
                
            b_res = torch.zeros((self.vocab_size, self.max_len - 1), device=device)
            # Create a one-hot like target for argmax
            for t, token_id in enumerate(best_seq):
                if token_id >= 0 and token_id < self.vocab_size:
                    b_res[token_id, t] = 100.0
                
            all_res.append(b_res.unsqueeze(0))
            
        result = torch.cat(all_res, dim=0)
        
        if return_attention:
            # Create correctly shaped dummy alphas to avoid crashing visualization grid calculations
            alpha_size = features.size(1)
            if self.attn_type == 'adaptive':
                alpha_size += 1
            dummy_alphas = [torch.zeros((batch_size, alpha_size), device='cpu') for _ in range(self.max_len - 1)]
            return result, dummy_alphas
            
        return result

    def generate_with_saliency(self, img: torch.Tensor):
        """
        Generates caption and returns saliency maps (input gradients) for early_fusion xLSTM.
        Args:
            img: (1, 3, H, W) input image tensor. Must have requires_grad=True or we will enable it.
        Returns:
            result: (1, vocab, T) predicted logits
            saliency_maps: list of (1, 1, H, W) gradients for each generated token
        """
        assert img.shape[0] == 1, "Saliency generation only supports batch size 1"
        assert self.attn_type == 'early_fusion', "Saliency is optimally designed for early_fusion (visual prompting)"
        
        img = img.clone().detach().requires_grad_(True)
        batch_size = 1
        device = img.device
        
        all_preds = []
        saliency_maps = []
        
        # We need to re-extract features but keep the computational graph attached to img
        features = self._extract_features(img)
        features = self.encoder_proj(features)
        
        # Initial hidden state setup
        if self.use_attention and self.attn_type != 'early_fusion':
            h0 = features.mean(dim=1)
        elif self.attn_type == 'early_fusion':
            h0 = None
        else:
            h0 = features
            
        if h0 is not None:
            hidden = h0.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
            if self.decoder_type == 'lstm':
                cell = torch.zeros_like(hidden)
                hidden = (hidden, cell)
            elif self.decoder_type == 'xlstm':
                img_token = h0.unsqueeze(1)
                _, hidden = self.decoder.step(img_token, state=None)
        else:
            hidden = None
            
        # Process visual prefix
        sep = self.visual_separator.expand(batch_size, -1, -1)
        full_prefix = torch.cat([features, sep], dim=1)
        for f_i in range(full_prefix.size(1)):
            inp_f = full_prefix[:, f_i:f_i+1, :]
            if self.decoder_type == 'xlstm':
                _, hidden = self.decoder.step(inp_f, state=hidden)
            else:
                _, hidden = self.decoder(inp_f, hidden)
                
        curr_token = torch.full((1,), self.sos_idx, device=device, dtype=torch.long)
        
        for step in range(self.max_len - 1):
            if img.grad is not None:
                img.grad.zero_()
                
            inp = self.embed(curr_token).unsqueeze(1)
            
            if self.decoder_type == 'xlstm':
                inp = self.decoder_proj_inp(inp)
                out, hidden = self.decoder.step(inp, state=hidden)
            else:
                out, hidden = self.decoder(inp, hidden)
                
            logits = self.proj(out.squeeze(1))
            all_preds.append(logits.unsqueeze(2))
            
            # Predict the most likely token
            pred_id = logits.argmax(dim=1)
            curr_token = pred_id
            
            # Backward pass for the predicted class
            score = logits[0, pred_id[0]]
            
            # Retain graph to allow multiple backwards through the visual prefix
            score.backward(retain_graph=True)
            
            if img.grad is not None:
                # Take absolute value and channel mean to get saliency
                saliency = img.grad.abs().mean(dim=1, keepdim=True) # (1, 1, H, W)
                saliency_maps.append(saliency.detach().cpu())
            
            if curr_token[0] == self.eos_idx:
                break
                
        result = torch.cat(all_preds, dim=2)
        return result, saliency_maps

    def generate_with_pseudo_attention_and_surprise(self, img: torch.Tensor):
        """
        Generates caption and returns pseudo-attention maps (cosine similarity) 
        and surprise maps (delta hidden state) for early_fusion xLSTM.
        Returns:
            result: (1, vocab, T) predicted logits
            pseudo_attn_maps: list of (1, 1, H, W) cosine similarities for each generated token
            surprise_map: (1, 1, H, W) magnitude of hidden state change for each visual patch
        """
        assert img.shape[0] == 1, "Generation only supports batch size 1"
        assert self.attn_type == 'early_fusion', "This is designed for early_fusion"
        
        batch_size = 1
        device = img.device
        
        all_preds = []
        pseudo_attn_maps = []
        visual_hiddens = []
        surprise_values = []
        
        features = self._extract_features(img)
        features = self.encoder_proj(features)
        
        if self.use_attention and self.attn_type != 'early_fusion':
            h0 = features.mean(dim=1)
        elif self.attn_type == 'early_fusion':
            h0 = None
        else:
            h0 = features
            
        if h0 is not None:
            hidden = h0.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
            if self.decoder_type == 'lstm':
                cell = torch.zeros_like(hidden)
                hidden = (hidden, cell)
            elif self.decoder_type == 'xlstm':
                img_token = h0.unsqueeze(1)
                _, hidden = self.decoder.step(img_token, state=None)
        else:
            hidden = None
            
        # Process visual prefix
        prev_h = None
        sep = self.visual_separator.expand(batch_size, -1, -1)
        full_prefix = torch.cat([features, sep], dim=1)
        
        for f_i in range(full_prefix.size(1)):
            inp_f = full_prefix[:, f_i:f_i+1, :]
            if self.decoder_type == 'xlstm':
                out, hidden = self.decoder.step(inp_f, state=hidden)
            else:
                out, hidden = self.decoder(inp_f, hidden)
            
            # Save visual hidden state and Surprise only for actual patches (not separator)
            if f_i < features.size(1):
                curr_h = out.squeeze(1) # (1, dec_dim)
                visual_hiddens.append(curr_h)
                
                # Calculate Surprise (delta h_t as a robust proxy for delta C_t)
                if prev_h is not None:
                    delta = torch.norm(curr_h - prev_h, p=2).item()
                else:
                    delta = torch.norm(curr_h, p=2).item()
                surprise_values.append(delta)
                prev_h = curr_h
            
        # Stack visual hiddens: (1, L, dec_dim)
        visual_hiddens_stack = torch.stack(visual_hiddens, dim=1)
        
        # Build surprise map: (1, 1, H, W)
        import math
        L = features.size(1)
        H_grid = int(math.sqrt(L))
        W_grid = H_grid
        surprise_tensor = torch.tensor(surprise_values, device=device).view(1, 1, H_grid, W_grid)
        
        curr_token = torch.full((1,), self.sos_idx, device=device, dtype=torch.long)
        
        for step in range(self.max_len - 1):
            inp = self.embed(curr_token).unsqueeze(1)
            
            if self.decoder_type == 'xlstm':
                inp = self.decoder_proj_inp(inp)
                out, hidden = self.decoder.step(inp, state=hidden)
            else:
                out, hidden = self.decoder(inp, hidden)
                
            logits = self.proj(out.squeeze(1))
            all_preds.append(logits.unsqueeze(2))
            
            # Predict
            pred_id = logits.argmax(dim=1)
            curr_token = pred_id
            
            # Cosine Similarity between word h_t and all visual hiddens
            word_h = out # (1, 1, dec_dim)
            cos_sim = torch.nn.functional.cosine_similarity(word_h, visual_hiddens_stack, dim=-1) # (1, L)
            pseudo_attn_map = cos_sim.view(1, 1, H_grid, W_grid)
            pseudo_attn_maps.append(pseudo_attn_map.detach().cpu())
            
            if curr_token[0] == self.eos_idx:
                break
                
        result = torch.cat(all_preds, dim=2)
        return result, pseudo_attn_maps, surprise_tensor.cpu()

# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder',      default='resnet18', choices=list(ENCODER_CONFIGS))
    parser.add_argument('--freeze',       action='store_true')
    parser.add_argument('--decoder_type', default='gru', choices=['gru', 'lstm', 'xlstm'])
    parser.add_argument('--decoder_dim',  type=int, default=512)
    parser.add_argument('--layers',       type=int, default=1)
    parser.add_argument('--embed_dim',    type=int, default=512)
    parser.add_argument('--attention',    choices=['soft', 'adaptive', 'early_fusion'], default=None, help='Attention type')
    parser.add_argument('--attn_dim',     type=int, default=256)
    args = parser.parse_args()

    model = ImageCaptioningModel(
        encoder_name=args.encoder,
        freeze_encoder=args.freeze,
        decoder_type=args.decoder_type,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.layers,
        embed_dim=args.embed_dim,
        attn_type=args.attention,
        attn_dim=args.attn_dim,
    ).to(DEVICE)

    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
    out   = model(dummy)
    print(
        f"Encoder: {args.encoder} | Decoder: {args.decoder_type} | "
        f"Attention: {args.attention} | Output shape: {out.shape}"
    )
    assert out.shape[0] == 2 and out.shape[1] == NUM_CHAR and out.shape[2] <= TEXT_MAX_LEN - 1
    print("Model forward pass successful.")

    if args.attention:
        out2, alphas = model(dummy, return_attention=True)
        L = alphas[0].shape[1]
        grid = int(math.sqrt(L))
        print(
            f"Attention inference: {len(alphas)} steps, "
            f"alpha shape {alphas[0].shape}, spatial grid {grid}×{grid}"
        )
        assert alphas[0].shape == (2, L)
        print("Attention return check passed.")
