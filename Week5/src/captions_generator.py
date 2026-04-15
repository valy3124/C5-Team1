import pandas as pd
import json
import random
import os

QUALITIES = [
    "harsh camera flash", "flash glare", "uneven indoor lighting", "underexposed",
    "washed out", "harsh shadows", "dimly lit", "fluorescent lighting", "off-center composition",
    "partially cropped", "tilted angle (or dutch angle)", "awkward framing", "extreme close-up",
    "asymmetrical composition", "unconventional perspective", "grainy image", "digital noise",
    "raw unedited photo", "casual smartphone snapshot", "low-tier camera sensor", "obstructed view"
]

def main():
    targets_path = "../embeddings/cluster_generation_targets.csv"
    train_clusters_path = "../embeddings/train_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d.csv"
    annotations_path = "../../Week3/dataset/VizWiz/annotations/train.json"
    output_path = "../embeddings/cluster_prompts.csv"

    train_avgs_path = "../embeddings/train_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_cluster_averages.csv"
    
    print("Loading generation targets...")
    targets_df = pd.read_csv(targets_path)
    
    # Check if there are any clusters to generate
    # We only process clusters where num_samples_to_generate > 0
    targets_df = targets_df[targets_df["num_samples_to_generate"] > 0]
    clusters_to_generate = set(targets_df["cluster"].tolist())
    
    if not clusters_to_generate:
        print("No clusters currently require generation. Exiting.")
        return

    print("Loading train clusters mapping...")
    train_df = pd.read_csv(train_clusters_path)
    
    print("Loading and calculating cluster statistics...")
    train_avgs_df = pd.read_csv(train_avgs_path)
    mean_count = train_avgs_df['count'].mean()
    cluster_to_count = dict(zip(train_avgs_df['cluster'], train_avgs_df['count']))

    print("Loading VizWiz training annotations...")
    with open(annotations_path, 'r') as f:
        anno_data = json.load(f)

    # Dictionary mapping image_id to filename
    print("Parsing annotations...")
    id_to_filename = {img["id"]: img["file_name"] for img in anno_data["images"]}
    
    # Dictionary mapping filename to a list of its ground-truth captions
    filename_to_captions = {}
    for ann in anno_data["annotations"]:
        caption_text = ann["caption"].strip()
        if caption_text == "Quality issues are too severe to recognize visual content.":
            continue
            
        img_id = ann["image_id"]
        if img_id in id_to_filename:
            filename = id_to_filename[img_id]
            if filename not in filename_to_captions:
                filename_to_captions[filename] = []
            filename_to_captions[filename].append(caption_text)

    print("Generating prompts for each cluster...")
    results = []
    
    # Set seed for reproducability across runs
    random.seed(42)
    
    for _, row in targets_df.iterrows():
        cluster_id = int(row["cluster"])
        num_gen = int(row["num_samples_to_generate"])
        
        cluster_count = cluster_to_count.get(cluster_id, 0)
        is_above_average = cluster_count > mean_count
        
        # Get all filenames belonging to this specific cluster
        cluster_files = train_df[train_df["cluster"] == cluster_id]["filename"].tolist()
        
        # Collect GT captions for these files based on cluster size
        all_captions = []
        for fname in cluster_files:
            if fname in filename_to_captions:
                caps = filename_to_captions[fname]
                if is_above_average:
                    # Only pick 1 random caption for this image to maximize caption diversity 
                    # by forcing it to pull from more unique images in large clusters.
                    all_captions.append(random.choice(caps))
                else:
                    # Cluster is small, pull ALL captions from this image to make sure we have enough data
                    all_captions.extend(caps)
                
        # The user requested no top 20 limitation and no shuffling constraint
        sampled_captions = all_captions
        
        # Sample 3 or 4 random qualities to append to this cluster's prompt
        num_qualities = random.choice([3, 4])
        selected_qualities = random.sample(QUALITIES, num_qualities)
        qualities_str = ", ".join(selected_qualities)
        
        prompt = (
            f"Generate {num_gen} diverse image captions with maximum 17 words each. "
            f"Integrate {num_qualities} of the following visual characteristics/qualities naturally into only 3 of the generated captions: {qualities_str}.\n\n"
            f"Here you have examples of correct captions belonging to this cluster to guide the semantic content of your new captions:\n"
        )
        
        for cap in sampled_captions:
            prompt += f"- {cap.strip()}\n"
            
        results.append({
            "cluster_id": cluster_id,
            "num_samples_to_generate": num_gen,
            "prompt": prompt
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(out_df)} prompts and saved them to {output_path}!")

if __name__ == "__main__":
    main()
