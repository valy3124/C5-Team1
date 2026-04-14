"""
clustering.py — Analyze VizWiz training images via CLIP embeddings + UMAP + clustering.

Usage:
    # Step 1: Extract embeddings (run once, saves to disk)
    python clustering.py --step extract --data_dir ../Week3/dataset/VizWiz/images/train

    # Step 2: Launch interactive Streamlit app (all params configurable in sidebar)
    streamlit run clustering.py -- --step visualize --embeddings_path embeddings/clip_embeddings.npz --data_dir ../Week3/dataset/VizWiz/images/train
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import clip
from PIL import Image
import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image
import umap
import hdbscan
from sklearn.cluster import KMeans
from transformers import BlipModel, BlipProcessor


# Step 1: Extract embeddings (CLIP or BLIP)
def extract_embeddings(data_dir: str, encoder: str = "clip", output_dir: str = "embeddings", batch_size: int = 64):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    if encoder == "clip":
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
 
        def encode_batch(images):
            batch_tensor = torch.stack([preprocess(img) for img in images]).to(device)
            with torch.no_grad():
                features = model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()
 
    elif encoder == "blip":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True).to(device)
        model.eval()
 
        def encode_batch(images):
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.vision_model(**inputs)
                features = outputs.pooler_output
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()
 
    else:
        raise ValueError(f"Unsupported encoder: {encoder}. Choose 'clip' or 'blip'.")
 
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([
        p for p in Path(data_dir).iterdir()
        if p.suffix.lower() in extensions
    ])
    print(f"Found {len(image_paths)} images in {data_dir}")
    print(f"Using encoder: {encoder}")
 
    all_embeddings = []
    all_filenames = []
    failed = []
 
    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Extracting {encoder.upper()} embeddings"):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        batch_names = []
 
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
                batch_names.append(p.name)
            except Exception as e:
                failed.append((p.name, str(e)))
 
        if not batch_images:
            continue
 
        features = encode_batch(batch_images)
        all_embeddings.append(features)
        all_filenames.extend(batch_names)
 
    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    filenames = np.array(all_filenames)
 
    save_path = os.path.join(output_dir, f"{encoder}_embeddings.npz")
    np.savez(save_path, embeddings=embeddings, filenames=filenames)
    print(f"Saved {len(filenames)} embeddings to {save_path}")
    if failed:
        print(f"Failed to process {len(failed)} images")
 
    return embeddings, filenames


# Step 2: Streamlit interactive app
def run_streamlit(embeddings_path: str, data_dir: str):
    """
    Interactive Streamlit app with configurable UMAP + clustering params.
    """

    st.set_page_config(layout="wide", page_title="VizWiz Cluster Explorer")
    st.title("VizWiz Training Set — CLIP Embedding Clusters")

    # Load embeddings (cached)
    @st.cache_data
    def load_embeddings(path):
        data = np.load(path)
        return data["embeddings"], data["filenames"]

    embeddings, filenames = load_embeddings(embeddings_path)
    st.sidebar.markdown(f"**{len(embeddings)} images loaded**")

    # UMAP parameters
    st.sidebar.header("UMAP parameters")

    n_neighbors = st.sidebar.slider(
        "n_neighbors", 5, 100, 15,
        help="Low → captures fine local structure. High → preserves more global structure."
    )
    min_dist = st.sidebar.slider(
        "min_dist", 0.0, 1.0, 0.1, step=0.05,
        help="Low → tighter, more separated clusters. High → more uniform spread."
    )
    umap_metric = st.sidebar.selectbox(
        "Distance metric", ["cosine", "euclidean", "correlation"], index=0,
        help="Cosine is generally best for CLIP embeddings."
    )
    n_components = st.sidebar.radio(
        "UMAP dimensions", [2, 3], index=0,
        help="3D can reveal structure hidden in 2D, but harder to interpret."
    )

    # Clustering parameters
    st.sidebar.header("Clustering parameters")

    cluster_method = st.sidebar.selectbox("Method", ["HDBSCAN", "KMeans"])

    if cluster_method == "HDBSCAN":
        min_cluster_size = st.sidebar.slider(
            "min_cluster_size", 5, 200, 15,
            help="Minimum points to form a cluster. Higher → fewer, larger clusters."
        )
        min_samples = st.sidebar.slider(
            "min_samples", 1, 50, 5,
            help="How conservative clustering is. Higher → more points labeled as noise."
        )
        cluster_selection = st.sidebar.selectbox(
            "cluster_selection_method", ["eom", "leaf"],
            help="'eom' = variable-density clusters. 'leaf' = many small homogeneous clusters."
        )
    else:
        n_clusters_km = st.sidebar.slider(
            "n_clusters", 2, 50, 10,
            help="Number of clusters for KMeans."
        )

    # Run button
    run_clicked = st.sidebar.button(
        "Run UMAP + Clustering", type="primary", use_container_width=True
    )

    # Cached computation functions
    @st.cache_data
    def compute_umap(_emb, n_neighbors, min_dist, metric, n_components):
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42,
        )
        return reducer.fit_transform(_emb)

    @st.cache_data
    def compute_hdbscan(_umap_2d, min_cluster_size, min_samples, cluster_selection):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=cluster_selection,
        )
        return clusterer.fit_predict(_umap_2d)

    @st.cache_data
    def compute_kmeans(_umap_2d, n_clusters):
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return km.fit_predict(_umap_2d)

    # State management
    if "umap_result" not in st.session_state:
        st.session_state.umap_result = None
        st.session_state.labels = None

    if run_clicked:
        with st.spinner("Running UMAP... (this can take a minute)"):
            st.session_state.umap_result = compute_umap(
                embeddings, n_neighbors, min_dist, umap_metric, n_components
            )

        with st.spinner("Clustering..."):
            if cluster_method == "HDBSCAN":
                st.session_state.labels = compute_hdbscan(
                    st.session_state.umap_result, min_cluster_size, min_samples, cluster_selection
                )
            else:
                st.session_state.labels = compute_kmeans(
                    st.session_state.umap_result, n_clusters_km
                )

    umap_result = st.session_state.umap_result
    labels = st.session_state.labels

    if umap_result is None:
        st.info("👈 Configure parameters in the sidebar and click **Run UMAP + Clustering** to start.")
        return

    # Results
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    df_dict = {
        "UMAP_1": umap_result[:, 0],
        "UMAP_2": umap_result[:, 1],
        "cluster": labels.astype(str),
        "filename": filenames,
    }
    if n_components == 3:
        df_dict["UMAP_3"] = umap_result[:, 2]

    df = pd.DataFrame(df_dict)

    st.sidebar.header("Results")
    st.sidebar.metric("Clusters found", n_clusters_found)
    if n_noise > 0:
        st.sidebar.metric("Noise points", f"{n_noise} ({n_noise / len(labels) * 100:.1f}%)")

    cluster_sizes = df["cluster"].value_counts().reset_index()
    cluster_sizes.columns = ["Cluster", "Count"]
    cluster_sizes["Pct"] = (cluster_sizes["Count"] / len(df) * 100).round(1)
    st.sidebar.dataframe(cluster_sizes, use_container_width=True, hide_index=True)

    st.sidebar.header("Filter display")
    all_clusters = sorted(df["cluster"].unique(), key=lambda x: int(x))
    selected_clusters = st.sidebar.multiselect(
        "Show clusters", all_clusters, default=all_clusters
    )
    df_filtered = df[df["cluster"].isin(selected_clusters)]

    if n_components == 3:
        fig = px.scatter_3d(
            df_filtered,
            x="UMAP_1", y="UMAP_2", z="UMAP_3",
            color="cluster",
            hover_data=["filename"],
            title=f"UMAP 3D — {n_clusters_found} clusters, {len(df_filtered)} points",
            height=700,
        )
        fig.update_traces(marker=dict(size=2, opacity=0.7))
    else:
        fig = px.scatter(
            df_filtered,
            x="UMAP_1", y="UMAP_2",
            color="cluster",
            hover_data=["filename"],
            title=f"UMAP 2D — {n_clusters_found} clusters, {len(df_filtered)} points",
            height=700,
        )
        fig.update_traces(marker=dict(size=3, opacity=0.7))

    fig.update_layout(legend_title_text="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    # Explore images
    st.subheader("Explore images by cluster")

    col1, col2 = st.columns([1, 1])

    with col1:
        browse_cluster = st.selectbox("Select cluster to browse", all_clusters)
        cluster_files = df[df["cluster"] == browse_cluster]["filename"].values

        n_per_page = st.slider("Images per page", 5, 50, 20)
        total_pages = max(1, (len(cluster_files) + n_per_page - 1) // n_per_page)
        page = st.number_input("Page", 1, total_pages, 1) - 1

        start = page * n_per_page
        end = min(start + n_per_page, len(cluster_files))
        st.write(f"Showing {start + 1}–{end} of {len(cluster_files)} images in cluster {browse_cluster}")

        cols = st.columns(5)
        for i, fname in enumerate(cluster_files[start:end]):
            img_path = os.path.join(data_dir, fname)
            if os.path.exists(img_path):
                with cols[i % 5]:
                    st.image(img_path, caption=fname, width=150)

    with col2:
        st.markdown("**Search by filename**")
        search = st.text_input("Filename (partial match)")
        if search:
            matches = df[df["filename"].str.contains(search, case=False)]
            st.write(f"Found {len(matches)} matches")
            for _, row in matches.head(10).iterrows():
                img_path = os.path.join(data_dir, row["filename"])
                if os.path.exists(img_path):
                    st.image(
                        img_path,
                        caption=f"{row['filename']} (cluster {row['cluster']})",
                        width=200,
                    )

    # Export
    st.sidebar.header("Export")
    if st.sidebar.button("Save results to disk"):
        os.makedirs("embeddings", exist_ok=True)
        save_path = "embeddings/cluster_results.npz"
        np.savez(
            save_path,
            umap_2d=umap_result,
            labels=labels,
            filenames=filenames,
            embeddings=embeddings,
        )
        st.sidebar.success(f"Saved to {save_path}")

    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        "Download clusters as CSV",
        csv,
        "cluster_assignments.csv",
        "text/csv",
        use_container_width=True,
    )


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image embeddings + UMAP + clustering for VizWiz")
    parser.add_argument("--step", type=str, required=True, choices=["extract", "visualize"],
                        help="'extract' = compute embeddings, 'visualize' = launch Streamlit app")
    parser.add_argument("--encoder", type=str, default="clip", choices=["clip", "blip"],
                        help="Encoder to use for embedding extraction")
    parser.add_argument("--data_dir", type=str, default="../../Week3/dataset/VizWiz/images/train",
                        help="Path to training images directory")
    parser.add_argument("--embeddings_path", type=str, default="embeddings/clip_embeddings.npz",
                        help="Path to saved embeddings file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="embeddings")
 
    args = parser.parse_args()
 
    if args.step == "extract":
        extract_embeddings(args.data_dir, args.encoder, args.output_dir, args.batch_size)
 
    elif args.step == "visualize":
        run_streamlit(args.embeddings_path, args.data_dir)