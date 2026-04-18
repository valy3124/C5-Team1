import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import argparse
import json
import textwrap
import matplotlib.colors as mcolors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_cluster_csv', default='../embeddings/val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_cluster_averages.csv')
    parser.add_argument('--aug_cluster_csv', default='../embeddings/val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_cluster_averages_augmented.csv')
    parser.add_argument('--base_sample_csv', default='../embeddings/val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_with_metrics.csv')
    parser.add_argument('--aug_sample_csv', default='../embeddings/val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_with_metrics_augmented.csv')
    parser.add_argument('--val_json', default='/ghome/group01/C5/dataset/VizWiz/annotations/val.json')
    parser.add_argument('--img_dir', default='/ghome/group01/C5/dataset/VizWiz/images/val')
    parser.add_argument('--out_dir', default='../analysis/augmentation_results')
    parser.add_argument('--cluster_id', default='all', help="Specific cluster to visualize samples for, or 'all'")
    return parser.parse_args()

def wrap_text(text, width=80):
    return "\n".join(textwrap.wrap(str(text), width))

def main():
    args = parse_args()
    
    os.makedirs(os.path.join(args.out_dir, 'plots'), exist_ok=True)
    
    if args.cluster_id == 'all':
        sample_dir = os.path.join(args.out_dir, 'samples', 'all')
    else:
        sample_dir = os.path.join(args.out_dir, 'samples', f'cluster_{args.cluster_id}')
        
    improved_dir = os.path.join(sample_dir, 'improved')
    worsened_dir = os.path.join(sample_dir, 'worsened')
    os.makedirs(improved_dir, exist_ok=True)
    os.makedirs(worsened_dir, exist_ok=True)
    
    # 1. Load GT Mapping
    print("Loading GT captions...")
    with open(args.val_json, 'r') as f:
        val_data = json.load(f)
        
    id_to_filename = {img['id']: img['file_name'] for img in val_data['images']}
    filename_to_gt = {}
    for ann in val_data['annotations']:
        fname = id_to_filename.get(ann['image_id'])
        if fname:
            if fname not in filename_to_gt:
                filename_to_gt[fname] = []
            filename_to_gt[fname].append(ann['caption'])
            
    # 2. Cluster Level Analysis
    print("Analyzing Clusters...")
    df_base = pd.read_csv(args.base_cluster_csv)
    df_aug = pd.read_csv(args.aug_cluster_csv)
    
    df_base.rename(columns={'METEOR': 'METEOR_base', 'count': 'count'}, inplace=True)
    df_aug.rename(columns={'METEOR': 'METEOR_aug'}, inplace=True)
    df_cluster = pd.merge(df_base[['cluster', 'METEOR_base', 'count']], df_aug[['cluster', 'METEOR_aug']], on='cluster')
    
    # Compute Delta
    df_cluster['METEOR_delta'] = df_cluster['METEOR_aug'] - df_cluster['METEOR_base']
    
    # Save CSV
    csv_out = os.path.join(args.out_dir, 'cluster_metrics_delta.csv')
    df_cluster.to_csv(csv_out, index=False)
    print(f"Saved cluster delta CSV to {csv_out}")
    
    # ==========================================
    # PLOT 1: METEOR Delta per Cluster
    # ==========================================
    plt.figure(figsize=(20, 8))
    df_cluster_sorted = df_cluster.sort_values('METEOR_delta', ascending=False)
    
    vmin = df_cluster_sorted['METEOR_delta'].min()
    vmax = df_cluster_sorted['METEOR_delta'].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('RdYlGn')
    colors = [cmap(norm(x)) for x in df_cluster_sorted['METEOR_delta']]
    
    # Using matplotlib directly fixes the seaborn hue sorting bug that scrambled the gradient
    bars = plt.bar(df_cluster_sorted['cluster'].astype(str), df_cluster_sorted['METEOR_delta'], color=colors)
    
    # Increase fonts for slide readability
    plt.title('METEOR Delta per Cluster (Augmented - Baseline)', fontsize=24, pad=15)
    plt.ylabel('METEOR Improvement', fontsize=18)
    plt.xlabel('Cluster ID', fontsize=18)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=14)
    
    ax = plt.gca()
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 3 != 0:
            label.set_visible(False)
            
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("METEOR Delta", fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'plots', 'meteor_delta_per_cluster.png'))
    plt.close()
    
# ==========================================
    # PLOT 2: Top 5 worst baseline clusters
    # ==========================================
    worst_5_base = df_cluster.sort_values('METEOR_base', ascending=True).head(5)
    df_melt = worst_5_base.melt(id_vars=['cluster'], value_vars=['METEOR_base', 'METEOR_aug'], var_name='Model', value_name='METEOR Score')
    df_melt['Model'] = df_melt['Model'].map({'METEOR_base': 'Baseline', 'METEOR_aug': 'Augmented'})
    
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=df_melt, x='cluster', y='METEOR Score', hue='Model', palette=['#3498db', '#f39c12'], order=worst_5_base['cluster'].astype(int))
    
    clusters = worst_5_base['cluster'].astype(int).tolist()
    
# We use [:10] to ensure we only loop over the 10 data bars and ignore the legend rectangles
    for i, p in enumerate(ax.patches[:10]):
        idx = i % 5  
        cluster_id = clusters[idx]
        row = worst_5_base[worst_5_base['cluster'] == cluster_id].iloc[0]
        
        y_val = p.get_height()
        x_val = p.get_x() + p.get_width() / 2.0
        
        if i < 5:  
            # Baseline labels (Just the score)
            ax.text(x_val, y_val + 0.5, f"{row['METEOR_base']:.1f}", ha='center', va='bottom', fontsize=14)
        else:
            # Augmented labels
            
            # 1. Draw the actual Augmented METEOR score right above the bar
            ax.text(x_val, y_val + 0.5, f"{row['METEOR_aug']:.1f}", ha='center', va='bottom', fontsize=14)
            
            # 2. Draw the Delta (+/- improvement) in a box higher up
            delta = row['METEOR_delta']
            color = 'green' if delta > 0 else 'red' if delta < 0 else 'black'
            sign = '+' if delta > 0 else ''
            
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5, alpha=0.9)
            
            # INCREASED OFFSET HERE: Changed y_val + 3.5 to y_val + 5.0
            ax.text(x_val, y_val + 5.0, f"{sign}{delta:.1f}", 
                    color=color, ha='center', va='bottom', 
                    fontsize=14, fontweight='bold', bbox=bbox_props)
            
    plt.title('Top 5 Worst Performing Clusters: Baseline vs Augmented', fontsize=22, pad=20)
    plt.ylabel('METEOR Score', fontsize=18)
    plt.xlabel('Cluster ID', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title='Model', fontsize=14, title_fontsize=16, loc='upper right')
    
    # Increased padding at the top of the chart to 15 so the higher boxes fit perfectly
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] + 15) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'plots', 'worst_5_clusters_improvement.png'))
    plt.close()
        
    # 3. Sample Level Analysis
    print("Analyzing Samples...")
    df_base_samp = pd.read_csv(args.base_sample_csv)
    df_aug_samp = pd.read_csv(args.aug_sample_csv)
    
    df_samp = pd.merge(df_base_samp[['filename', 'cluster', 'METEOR', 'PREDICTION']], 
                       df_aug_samp[['filename', 'METEOR', 'PREDICTION']], 
                       on='filename', suffixes=('_base', '_aug'))
    
    if args.cluster_id != 'all':
        try:
            cid = int(args.cluster_id)
            df_samp = df_samp[df_samp['cluster'] == cid]
            print(f"Filtering samples for cluster {cid}. Found {len(df_samp)} images.")
        except ValueError:
            print("Invalid cluster_id.")
    else:
        df_samp = df_samp[df_samp['cluster'] != -1]
        
    df_samp['METEOR_delta'] = df_samp['METEOR_aug'] - df_samp['METEOR_base']
    df_samp['PREDICTION_base'] = df_samp['PREDICTION_base'].fillna('')
    df_samp['PREDICTION_aug'] = df_samp['PREDICTION_aug'].fillna('')
    df_samp_sorted = df_samp.dropna(subset=['METEOR_delta']).sort_values('METEOR_delta', ascending=False)
    
    def save_sample_plots(df_subset, save_dir, prefix):
        for idx, row in enumerate(df_subset.iterrows()):
            _, row = row
            img_path = os.path.join(args.img_dir, row['filename'])
            if os.path.exists(img_path):
                img = Image.open(img_path)
                
                gt_list = filename_to_gt.get(row['filename'], ["No GT found"])
                gts = " | ".join(gt_list[:2])
                
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')
                title = f"{prefix} #{idx+1} (Cluster: {row['cluster']})\n"
                title += f"Base METEOR: {row['METEOR_base']:.2f} | Aug METEOR: {row['METEOR_aug']:.2f}\n"
                title += wrap_text(f"GT: {gts}", 100) + "\n"
                title += wrap_text(f"Base Pred: {row['PREDICTION_base']}", 100) + "\n"
                title += wrap_text(f"Aug Pred: {row['PREDICTION_aug']}", 100)
                plt.title(title, loc='left', pad=10, fontsize=12)
                
                out_path = os.path.join(save_dir, f"{prefix.lower().replace(' ', '_')}_{idx+1}_{row['filename']}")
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
            else:
                print(f"Image not found at {img_path}")
                
    top_improved = df_samp_sorted.head(5)
    top_worsened = df_samp_sorted.tail(5).iloc[::-1]
    
    print("Saving top 5 improved samples...")
    save_sample_plots(top_improved, improved_dir, "Improved")
    
    print("Saving top 5 worsened samples...")
    save_sample_plots(top_worsened, worsened_dir, "Worsened")
    
    print(f"Done! Results saved in {args.out_dir}")

if __name__ == "__main__":
    main()