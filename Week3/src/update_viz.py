import os
import re

with open('/ghome/group01/C5/xavi/C5-Team1/Week3/train.py', 'r') as f:
    text = f.read()

start_str = "            # Decode tokens (stop at EOS)"
end_str = "            # ----------------------------------------------------------\n            #  Training / evaluation loops"

start_idx = text.find(start_str)
end_idx = text.find(end_str)

if start_idx == -1 or end_idx == -1:
    print("Could not find blocks")
    exit(1)

new_code = """            # Decode tokens (stop at EOS)
            tokens = []
            valid_attn_maps = []
            words = []
            for t, idx in enumerate(pred_indices):
                idx_val = idx.item()
                if idx_val == getattr(tokenizer, 'eos_idx', -1):
                    break
                # Skip padding, sos, and unk tokens for better visualization
                if idx_val in [getattr(tokenizer, 'pad_idx', -1), getattr(tokenizer, 'sos_idx', -1)]:
                    continue
                if hasattr(tokenizer, 'unk_idx') and idx_val == tokenizer.unk_idx:
                    continue
                    
                word = tokenizer.decode([idx_val]).strip() or f"tok{idx_val}"
                if not word or word == "<UNK>" or word == "[UNK]":
                    continue
                    
                tokens.append(idx_val)
                words.append(word)
                if t < len(attn_maps):
                    valid_attn_maps.append(attn_maps[t])

            if not tokens:
                print(f"  [skip] {img_name} — empty prediction after filtering.")
                continue

            n_words = min(len(tokens), len(valid_attn_maps))

            # ----------------------------------------------------------
            # Per-word individual images (for soft / adaptive)
            # ----------------------------------------------------------
            img_stem    = os.path.splitext(img_name)[0]
            img_out_dir = os.path.join(attn_dir, f"{img_stem}_outputs")
            os.makedirs(img_out_dir, exist_ok=True)

            if model.attn_type in ['soft', 'adaptive']:
                for t in range(n_words):
                    word    = words[t]
                    alpha   = valid_attn_maps[t][0][:L].numpy()              # (L,)
                    alpha_grid = alpha.reshape(grid_h, grid_w)
                    alpha_pil  = PILImage.fromarray((alpha_grid * 255).astype('uint8')).resize((orig_w, orig_h), resample=PILImage.BILINEAR)
                    alpha_arr  = np.array(alpha_pil) / 255.0
                    alpha_norm = (alpha_arr - alpha_arr.min()) / (alpha_arr.max() - alpha_arr.min() + 1e-8)
                    
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    if model.attn_type == 'adaptive':
                        beta_t = valid_attn_maps[t][0][-1].item()
                        vis_prob = 1.0 - beta_t
                        # Fade the brightness based on how much it attends to the image
                        display_img = (orig_arr * (alpha_norm[..., np.newaxis] * vis_prob * 0.9 + 0.1)).astype(np.uint8)
                        ax.imshow(display_img)
                        ax.set_title(f'"{word}" (p_img={vis_prob:.2f})', fontsize=11, fontweight='bold')
                    else:
                        display_img = (orig_arr * (alpha_norm[..., np.newaxis] * 0.8 + 0.2)).astype(np.uint8)
                        ax.imshow(display_img)
                        ax.set_title(f'"{word}"', fontsize=13, fontweight='bold')
                        
                    ax.axis('off')
                    plt.tight_layout(pad=0.3)
                    safe_word = "".join(c if c.isalnum() else "_" for c in word)[:20]
                    save_path = os.path.join(img_out_dir, f"word_{t:02d}_{safe_word}.png")
                    plt.savefig(save_path, dpi=90, bbox_inches='tight')
                    plt.close(fig)

            # ----------------------------------------------------------
            # Composite grids / multi-color visualization
            # ----------------------------------------------------------
            if model.attn_type == 'soft':
                n_cells  = n_words + 1
                n_cols   = min(5, n_cells)
                n_rows   = math.ceil(n_cells / n_cols)

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 3.0))
                axes = np.array(axes).reshape(-1)

                axes[0].imshow(orig_arr)
                axes[0].set_title("Input", fontsize=10, fontweight='bold')
                axes[0].axis('off')

                for t in range(n_words):
                    word    = words[t]
                    alpha   = valid_attn_maps[t][0][:L].numpy()
                    alpha_grid = alpha.reshape(grid_h, grid_w)
                    alpha_pil  = PILImage.fromarray((alpha_grid * 255).astype('uint8')).resize((orig_w, orig_h), resample=PILImage.BILINEAR)
                    alpha_arr  = np.array(alpha_pil) / 255.0
                    alpha_norm = (alpha_arr - alpha_arr.min()) / (alpha_arr.max() - alpha_arr.min() + 1e-8)

                    ax = axes[t + 1]
                    display_img = (orig_arr * (alpha_norm[..., np.newaxis] * 0.8 + 0.2)).astype(np.uint8)
                    ax.imshow(display_img)
                    ax.set_title(f'"{word}"', fontsize=9)
                    ax.axis('off')

                for ax in axes[n_words + 1:]:
                    ax.axis('off')

                pred_display = pred_str[:80] + ("…" if len(pred_str) > 80 else "")
                plt.suptitle(f"Pred: {pred_display}", fontsize=8, y=1.01)
                plt.tight_layout(pad=0.5)

                grid_path = os.path.join(img_out_dir, "attention_grid.png")
                plt.savefig(grid_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {img_out_dir}/")
                
            elif model.attn_type == 'adaptive':
                # Multicolor text overlay
                import matplotlib.colors as mcolors
                colors_list = list(mcolors.TABLEAU_COLORS.values())
                if len(colors_list) < n_words:
                    colors_list = colors_list * (n_words // len(colors_list) + 1)
                
                fig2 = plt.figure(figsize=(6, 6))
                ax_img2 = plt.axes([0.1, 0.3, 0.8, 0.65])
                ax_text2 = plt.axes([0.1, 0.05, 0.8, 0.2])
                ax_img2.imshow(orig_arr)
                ax_img2.axis('off')
                
                overlay = np.zeros((*orig_arr.shape[:2], 4))
                current_x, max_x, y = 0.0, 1.0, 0.8
                
                for t in range(n_words):
                    word = words[t]
                    alpha = valid_attn_maps[t][0][:L].numpy()
                    alpha_grid = alpha.reshape(grid_h, grid_w)
                    alpha_pil = PILImage.fromarray((alpha_grid * 255).astype('uint8')).resize((orig_w, orig_h), resample=PILImage.BILINEAR)
                    alpha_arr = np.array(alpha_pil) / 255.0
                    alpha_norm = (alpha_arr - alpha_arr.min()) / (alpha_arr.max() - alpha_arr.min() + 1e-8)
                    vis_prob = 1.0 - valid_attn_maps[t][0][-1].item()
                    
                    color_hex = colors_list[t]
                    c = mcolors.to_rgb(color_hex)
                    # Blend where attention is strong and visual sentinel is low (high vis_prob)
                    mask = (alpha_norm > np.percentile(alpha_arr, 85)) & (vis_prob > 0.4)
                    
                    for i in range(3):
                        overlay[..., i] = np.where(mask, c[i], overlay[..., i])
                    overlay[..., 3] = np.where(mask, np.maximum(overlay[..., 3], alpha_norm * 0.6), overlay[..., 3])
                    
                    text_obj = ax_text2.text(current_x, y, word, fontsize=14, color='black')
                    approx_width = max(len(word), 2) * 0.03
                    if current_x + approx_width > max_x:
                        y -= 0.3
                        current_x = 0.0
                        text_obj = ax_text2.text(current_x, y, word, fontsize=14, color='black')
                    if vis_prob > 0.4:
                        ax_text2.plot([current_x, current_x + approx_width * 0.8], [y - 0.1, y - 0.1], linewidth=4, color=color_hex)
                    current_x += approx_width
                
                ax_img2.imshow(overlay)
                ax_text2.axis('off')
                multicolor_path = os.path.join(img_out_dir, "adaptive_multicolor.png")
                plt.savefig(multicolor_path, dpi=120, bbox_inches='tight')
                plt.close(fig2)

                # Line graph of visual grounding probabilities
                fig3 = plt.figure(figsize=(max(10, n_words * 1.5), 5))
                import matplotlib.gridspec as gridspec
                gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2.5], height_ratios=[2, 1.2], figure=fig3)
                
                ax_orig = fig3.add_subplot(gs[:, 0])
                ax_orig.imshow(orig_arr)
                ax_orig.axis('off')
                
                ax_line = fig3.add_subplot(gs[0, 1])
                vis_probs = [(1.0 - valid_attn_maps[t][0][-1].item()) for t in range(n_words)]
                ax_line.plot(range(n_words), vis_probs, marker='o', linestyle='-', linewidth=2, markersize=8, color='#4A90E2')
                ax_line.set_ylim(-0.05, 1.05)
                ax_line.set_xlim(-0.5, n_words - 0.5)
                ax_line.set_xticks([])
                ax_line.grid(True, axis='y', linestyle='--', alpha=0.7)
                for i, p in enumerate(vis_probs):
                    ax_line.text(i, p + 0.05 if p < 0.8 else p - 0.15, f"{p:.3f}", ha='center', va='bottom', fontsize=10)
                
                gs_thumbs = gridspec.GridSpecFromSubplotSpec(1, n_words, subplot_spec=gs[1, 1], wspace=0.1)
                import matplotlib.cm as cm
                jet = cm.get_cmap('jet')
                for t in range(n_words):
                    ax_thumb = fig3.add_subplot(gs_thumbs[0, t])
                    alpha = valid_attn_maps[t][0][:L].numpy()
                    alpha_grid = alpha.reshape(grid_h, grid_w)
                    alpha_pil = PILImage.fromarray((alpha_grid * 255).astype('uint8')).resize((orig_w, orig_h), resample=PILImage.BILINEAR)
                    alpha_arr = np.array(alpha_pil) / 255.0
                    alpha_norm = (alpha_arr - alpha_arr.min()) / (alpha_arr.max() - alpha_arr.min() + 1e-8)
                    
                    colored_alpha = jet(alpha_norm)[..., :3]
                    # To mimic the paper thumbnail: original image overlaid with jet heatmap
                    blended = (orig_arr/255.0 * 0.5 + colored_alpha * 0.5)
                    blended = np.clip(blended, 0, 1)
                    
                    ax_thumb.imshow(blended)
                    ax_thumb.axis('off')
                    ax_thumb.set_title(words[t], y=-0.4, fontsize=12)
                
                plt.subplots_adjust(wspace=0.05, hspace=0)
                sentinel_path = os.path.join(img_out_dir, "adaptive_sentinel_plot.png")
                plt.savefig(sentinel_path, dpi=120, bbox_inches='tight')
                plt.close(fig3)
                
                print(f"  Saved Adaptive plots: {img_out_dir}/")
"""

new_text = text[:start_idx] + new_code + "\n" + text[end_idx:]

with open('/ghome/group01/C5/xavi/C5-Team1/Week3/train.py', 'w') as f:
    f.write(new_text)

print("success")
