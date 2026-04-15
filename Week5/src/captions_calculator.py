import pandas as pd
import numpy as np
import os

def calculate_samples_to_generate(
    csv_path: str,
    output_path: str,
    min_samples: int = 5,
    max_samples: int = 80
):
    """
    Reads the cluster averages CSV, extracts the METEOR score for each cluster,
    and applies an inverted min-max formula to determine how many new samples to generate.
    Lower METEOR score -> More samples generated.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find the CSV file at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Initialize all to 0
    df['num_samples_to_generate'] = 0
    
    # Define threshold cutoff
    meteor_threshold = 39.82
    mask = df['METEOR'] < meteor_threshold
    
    if mask.sum() > 0:
        target_meteors = df.loc[mask, 'METEOR']
        
        min_meteor = target_meteors.min()
        max_meteor = meteor_threshold  # Mapping starts exactly here
        
        # Inverted score: 
        # 1.0 means it has the absolute lowest METEOR (worst performance)
        # 0.0 means it has the threshold METEOR
        inverted_normalized = 1.0 - ((target_meteors - min_meteor) / (max_meteor - min_meteor + 1e-6))
        inverted_normalized = np.clip(inverted_normalized, 0.0, 1.0)
        
        # Map this 0-1 scale to our min_samples - max_samples range
        num_samples = min_samples + inverted_normalized * (max_samples - min_samples)
        
        # Convert to integer and apply to dataframe
        df.loc[mask, 'num_samples_to_generate'] = np.round(num_samples).astype(int)
    
    # Filter only the columns we want for the output
    out_df = df[['cluster', 'METEOR', 'num_samples_to_generate']].copy()
    out_df.rename(columns={'METEOR': 'average_meteor'}, inplace=True)
    
    # Split rows where num_samples_to_generate > 50
    split_rows = []
    for _, row in out_df.iterrows():
        samples = row['num_samples_to_generate']
        if samples > 50:
            half1 = samples // 2
            half2 = samples - half1
            
            row1 = row.copy()
            row1['num_samples_to_generate'] = half1
            split_rows.append(row1)
            
            row2 = row.copy()
            row2['num_samples_to_generate'] = half2
            split_rows.append(row2)
        else:
            split_rows.append(row)
            
    out_df = pd.DataFrame(split_rows)
    
    # Save the dataframe
    out_df.to_csv(output_path, index=False)
    print(f"Successfully calculated generation targets for {len(out_df)} clusters.")
    print(f"Total new samples to generate across all clusters: {out_df['num_samples_to_generate'].sum()}")
    print(f"Saved generation targets to: {output_path}")
    
    # Display some stats
    print("\nTop 5 clusters needing MOST samples (Worst METEOR):")
    print(out_df.sort_values('num_samples_to_generate', ascending=False).head(5))
    
    print("\nTop 5 clusters needing LEAST samples (Best METEOR):")
    print(out_df.sort_values('num_samples_to_generate', ascending=True).head(5))
    
    return out_df

if __name__ == "__main__":
    # Define paths
    EMBEDDINGS_DIR = "../embeddings"
    INPUT_CSV = os.path.join(EMBEDDINGS_DIR, "val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_cluster_averages.csv")
    OUTPUT_CSV = os.path.join(EMBEDDINGS_DIR, "cluster_generation_targets.csv")
    
    # Define generation boundaries
    MIN_SAMPLES_PER_CLUSTER = 15 # The absolute best performing cluster will generate 5 samples
    MAX_SAMPLES_PER_CLUSTER = 100 # The absolute worst performing cluster will generate 80 samples
    
    calculate_samples_to_generate(
        csv_path=INPUT_CSV,
        output_path=OUTPUT_CSV,
        min_samples=MIN_SAMPLES_PER_CLUSTER,
        max_samples=MAX_SAMPLES_PER_CLUSTER
    )