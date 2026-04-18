import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Analysis of Model Augmentation\nThis notebook analyzes the performance differences between the baseline model and the data-augmented model."),
    nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import matplotlib.image as mpimg

# Define paths
baseline_cluster_csv = '../embeddings/val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_cluster_averages.csv'
augmented_cluster_csv = '../embeddings/val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_cluster_averages_augmented.csv'

baseline_sample_csv = '../embeddings/val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_with_metrics.csv'
augmented_sample_csv = '../embeddings/val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_with_metrics_augmented.csv'

img_dir = '/ghome/group01/C5/dataset/VizWiz/images/val'
"""),
    nbf.v4.new_markdown_cell("## Data Loading\nLoading cluster averages."),
    nbf.v4.new_code_cell("""
df_base = pd.read_csv(baseline_cluster_csv)
df_aug = pd.read_csv(augmented_cluster_csv)

df_base.rename(columns={'METEOR': 'METEOR_base', 'count': 'count'}, inplace=True)
df_aug.rename(columns={'METEOR': 'METEOR_aug'}, inplace=True)

df_cluster = pd.merge(df_base[['cluster', 'METEOR_base', 'count']], df_aug[['cluster', 'METEOR_aug']], on='cluster')
df_cluster['METEOR_delta'] = df_cluster['METEOR_aug'] - df_cluster['METEOR_base']

# Print overview
print("Total clusters:", len(df_cluster))
print("Clusters that improved:", len(df_cluster[df_cluster['METEOR_delta'] > 0]))
print("Clusters that worsened:", len(df_cluster[df_cluster['METEOR_delta'] < 0]))
"""),
    nbf.v4.new_markdown_cell("## Cluster Improvement Analysis\nBar plot showing improvement and regression per cluster."),
    nbf.v4.new_code_cell("""
plt.figure(figsize=(15, 6))
df_cluster_sorted = df_cluster.sort_values('METEOR_delta', ascending=False)
colors = ['green' if x > 0 else 'red' for x in df_cluster_sorted['METEOR_delta']]
sns.barplot(data=df_cluster_sorted, x='cluster', y='METEOR_delta', palette=colors, order=df_cluster_sorted['cluster'])
plt.xticks(rotation=90)
plt.title('METEOR Delta per Cluster (Augmented - Baseline)')
plt.ylabel('METEOR Improvement')
plt.xlabel('Cluster ID')
plt.tight_layout()
plt.show()
"""),
    nbf.v4.new_markdown_cell("## Cluster Size Effect\nDoes the number of samples in a cluster correlate with the improvement?"),
    nbf.v4.new_code_cell("""
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_cluster, x='count', y='METEOR_delta')
plt.title('METEOR Improvement vs Cluster Size')
plt.xlabel('Number of Validation Samples in Cluster')
plt.ylabel('METEOR Delta')
plt.axhline(0, color='red', linestyle='--')
plt.tight_layout()
plt.show()
"""),
    nbf.v4.new_markdown_cell("## Sample Examination\nFinding the samples that improved or worsened the most."),
    nbf.v4.new_code_cell("""
df_base_samp = pd.read_csv(baseline_sample_csv)
df_aug_samp = pd.read_csv(augmented_sample_csv)

df_samp = pd.merge(df_base_samp[['filename', 'cluster', 'METEOR', 'PREDICTION']], 
                   df_aug_samp[['filename', 'METEOR', 'PREDICTION']], 
                   on='filename', suffixes=('_base', '_aug'))

df_samp['METEOR_delta'] = df_samp['METEOR_aug'] - df_samp['METEOR_base']
df_samp_sorted = df_samp.sort_values('METEOR_delta', ascending=False).dropna()
"""),
    nbf.v4.new_code_cell("""
def plot_samples(df_subset, title_prefix):
    for idx, row in df_subset.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')
            title = f"{title_prefix}\\nCluster: {row['cluster']} | Base METEOR: {row['METEOR_base']:.2f} | Aug METEOR: {row['METEOR_aug']:.2f}\\n"
            title += f"Base Pred: {row['PREDICTION_base']}\\n"
            title += f"Aug Pred: {row['PREDICTION_aug']}"
            plt.title(title, loc='left', pad=10)
            plt.show()
        else:
            print(f"Image not found at {img_path}")
"""),
    nbf.v4.new_markdown_cell("### Most Improved Samples"),
    nbf.v4.new_code_cell("""
top_improved = df_samp_sorted.head(5)
plot_samples(top_improved, "Most Improved Sample")
"""),
    nbf.v4.new_markdown_cell("### Most Worsened Samples"),
    nbf.v4.new_code_cell("""
top_worsened = df_samp_sorted.tail(5).iloc[::-1]  # Reverse to show worst first
plot_samples(top_worsened, "Most Worsened Sample")
""")
]

# Create notebooks directory if it doesn't exist
os.makedirs('/ghome/group01/C5/benet/C5-Team1/Week5/notebooks', exist_ok=True)

with open('/ghome/group01/C5/benet/C5-Team1/Week5/notebooks/augmentation_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated at /ghome/group01/C5/benet/C5-Team1/Week5/notebooks/augmentation_analysis.ipynb")
