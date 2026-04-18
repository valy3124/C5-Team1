"""
clustering.py — Analyze VizWiz training images via CLIP embeddings + UMAP + clustering.

Usage:
    From inside the src folder:

    # Step 1: Extract embeddings (run once, saves to disk)
    python clustering.py --step extract --data_dir /ghome/group01/C5/dataset/VizWiz/images/train --val_data_dir /ghome/group01/C5/dataset/VizWiz/images/val
    - CLIP: python clustering.py --step extract --data_dir /ghome/group01/C5/dataset/VizWiz/images/train --val_data_dir /ghome/group01/C5/dataset/VizWiz/images/val
    - BLIP: python clustering.py --step extract --data_dir /ghome/group01/C5/dataset/VizWiz/images/train --val_data_dir /ghome/group01/C5/dataset/VizWiz/images/val --encoder blip

    # Step 2: Launch interactive Streamlit app (all params configurable in sidebar)
    streamlit run clustering.py -- --step visualize --embeddings_path clip_embeddings/clip_embeddings_train.npz --data_dir /ghome/group01/C5/dataset/VizWiz/images/train
    - With images: streamlit run clustering.py -- --step visualize --embeddings_path ../embeddings/clip_embeddings_train.npz --val_embeddings_path ../embeddings/clip_embeddings_val.npz --data_dir /ghome/group01/C5/dataset/VizWiz/images/train --val_data_dir /ghome/group01/C5/dataset/VizWiz/images/val
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
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import umap
import hdbscan
from sklearn.cluster import KMeans
from transformers import BlipModel, BlipProcessor


# Step 1: Extract embeddings (CLIP or BLIP)
def extract_embeddings(data_dir: str, val_data_dir: str = None, encoder: str = "clip", output_dir: str = "../embeddings", batch_size: int = 64):
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
    
    def process_directory(directory: str, split_name: str):
        image_paths = sorted([
            p for p in Path(directory).iterdir()
            if p.suffix.lower() in extensions
        ])
        print(f"Found {len(image_paths)} images in {directory} ({split_name})")
        print(f"Using encoder: {encoder}")
     
        all_embeddings = []
        all_filenames = []
        failed = []
     
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Extracting {encoder.upper()} {split_name} embeddings"):
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
     
        save_path = os.path.join(output_dir, f"{encoder}_embeddings_{split_name}.npz")
        np.savez(save_path, embeddings=embeddings, filenames=filenames)
        print(f"Saved {len(filenames)} embeddings to {save_path}")
        if failed:
            print(f"Failed to process {len(failed)} images")
     
        return embeddings, filenames

    train_embeddings, train_filenames = process_directory(data_dir, "train")
    if val_data_dir:
        val_embeddings, val_filenames = process_directory(val_data_dir, "val")
        return (train_embeddings, train_filenames), (val_embeddings, val_filenames)

    return train_embeddings, train_filenames


# Step 2: Streamlit interactive app
def run_streamlit(embeddings_path: str, val_embeddings_path: str, data_dir: str, val_data_dir: str):
    """
    Interactive Streamlit app with configurable UMAP + clustering params.
    """

    st.set_page_config(layout="wide", page_title="VizWiz Cluster Explorer")
    st.title("VizWiz Train/Val Set — Embedding Clusters")

    # Load embeddings (cached)
    @st.cache_data
    def load_embeddings(path):
        data = np.load(path)
        return data["embeddings"], data["filenames"]

    embeddings, filenames = load_embeddings(embeddings_path)
    st.sidebar.markdown(f"**{len(embeddings)} train images loaded**")
    
    val_embeddings, val_filenames = None, None
    if val_embeddings_path and os.path.exists(val_embeddings_path):
        val_embeddings, val_filenames = load_embeddings(val_embeddings_path)
        st.sidebar.markdown(f"**{len(val_embeddings)} validation images loaded**")

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
    def compute_umap(_train_emb, _val_emb, n_neighbors, min_dist, metric, n_components):
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42,
        )
        train_umap = reducer.fit_transform(_train_emb)
        val_umap = reducer.transform(_val_emb) if _val_emb is not None else None
        return train_umap, val_umap

    @st.cache_data
    def compute_hdbscan(_train_umap, _val_umap, min_cluster_size, min_samples, cluster_selection):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=cluster_selection,
            prediction_data=True
        )
        train_labels = clusterer.fit_predict(_train_umap)
        val_labels = None
        if _val_umap is not None:
            val_labels, _ = hdbscan.approximate_predict(clusterer, _val_umap)
        return train_labels, val_labels

    @st.cache_data
    def compute_kmeans(_train_umap, _val_umap, n_clusters):
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        train_labels = km.fit_predict(_train_umap)
        val_labels = km.predict(_val_umap) if _val_umap is not None else None
        return train_labels, val_labels

    # State management
    if "train_umap_result" not in st.session_state:
        st.session_state.train_umap_result = None
        st.session_state.val_umap_result = None
        st.session_state.train_labels = None
        st.session_state.val_labels = None

    if run_clicked:
        with st.spinner("Running UMAP... (this can take a minute)"):
            t_umap, v_umap = compute_umap(
                embeddings, val_embeddings, n_neighbors, min_dist, umap_metric, n_components
            )
            st.session_state.train_umap_result = t_umap
            st.session_state.val_umap_result = v_umap

        with st.spinner("Clustering..."):
            if cluster_method == "HDBSCAN":
                t_lbl, v_lbl = compute_hdbscan(
                    t_umap, v_umap, min_cluster_size, min_samples, cluster_selection
                )
            else:
                t_lbl, v_lbl = compute_kmeans(
                    t_umap, v_umap, n_clusters_km
                )
            st.session_state.train_labels = t_lbl
            st.session_state.val_labels = v_lbl

    train_umap_result = st.session_state.train_umap_result
    val_umap_result = st.session_state.val_umap_result
    train_labels = st.session_state.train_labels
    val_labels = st.session_state.val_labels

    umap_result = train_umap_result
    labels = train_labels

    if umap_result is None:
        st.info("👈 Configure parameters in the sidebar and click **Run UMAP + Clustering** to start.")
        return

    # Results
    n_clusters_found = len(set(train_labels)) - (1 if -1 in train_labels else 0)
    n_noise = (train_labels == -1).sum()

    df_dict_train = {
        "UMAP_1": train_umap_result[:, 0],
        "UMAP_2": train_umap_result[:, 1],
        "cluster": train_labels.astype(str),
        "filename": filenames,
        "split": ["train"] * len(filenames)
    }
    if n_components == 3:
        df_dict_train["UMAP_3"] = train_umap_result[:, 2]

    df_train = pd.DataFrame(df_dict_train)
    
    if val_umap_result is not None:
        df_dict_val = {
            "UMAP_1": val_umap_result[:, 0],
            "UMAP_2": val_umap_result[:, 1],
            "cluster": val_labels.astype(str),
            "filename": val_filenames,
            "split": ["val"] * len(val_filenames)
        }
        if n_components == 3:
            df_dict_val["UMAP_3"] = val_umap_result[:, 2]
        df_val = pd.DataFrame(df_dict_val)
        df = pd.concat([df_train, df_val], ignore_index=True)
    else:
        df = df_train

    st.sidebar.header("Results")
    st.sidebar.metric("Clusters found", n_clusters_found)
    if n_noise > 0:
        st.sidebar.metric("Train Noise points", f"{n_noise} ({n_noise / len(train_labels) * 100:.1f}%)")

    cluster_sizes = df.groupby(["cluster", "split"]).size().unstack(fill_value=0).reset_index()
    if "val" not in cluster_sizes.columns:
        cluster_sizes.rename(columns={"train": "Train Samples"}, inplace=True)
    elif "train" not in cluster_sizes.columns:
        cluster_sizes.rename(columns={"val": "Val Samples"}, inplace=True)
    else:
        cluster_sizes.rename(columns={"train": "Train Samples", "val": "Val Samples"}, inplace=True)
        # Add total count
        cluster_sizes["Total"] = cluster_sizes["Train Samples"] + cluster_sizes["Val Samples"]
        
    st.sidebar.subheader("Cluster Samples Distribution")
    st.sidebar.dataframe(cluster_sizes, use_container_width=True, hide_index=True)

    st.sidebar.header("Filter display")
    all_clusters = sorted(df["cluster"].unique(), key=lambda x: int(x))
    selected_clusters = st.sidebar.multiselect(
        "Show clusters", all_clusters, default=all_clusters
    )
    
    show_val = False
    if "val" in df["split"].unique():
        show_val = st.sidebar.toggle("Toggle Validation Samples", value=True)
    
    selected_splits = ["train", "val"] if show_val else ["train"]
    df_filtered = df[(df["cluster"].isin(selected_clusters)) & (df["split"].isin(selected_splits))]

    if n_components == 3:
        fig = px.scatter_3d(
            df_filtered,
            x="UMAP_1", y="UMAP_2", z="UMAP_3",
            color="cluster",
            symbol="split",
            hover_data=["filename", "split"],
            title=f"UMAP 3D — {n_clusters_found} clusters, {len(df_filtered)} points",
            height=700,
        )
        fig.for_each_trace(lambda t: t.update(
            marker=dict(
                size=4 if t.name and 'val' in str(t.name).lower() else 2,
                opacity=0.25 if t.name and 'val' in str(t.name).lower() else 1.0
            )
        ))
    else:
        fig = px.scatter(
            df_filtered,
            x="UMAP_1", y="UMAP_2",
            color="cluster",
            symbol="split",
            hover_data=["filename", "split"],
            title=f"UMAP 2D — {n_clusters_found} clusters, {len(df_filtered)} points",
            height=700,
        )
        fig.for_each_trace(lambda t: t.update(
            marker=dict(
                size=5 if t.name and 'val' in str(t.name).lower() else 3,
                opacity=0.25 if t.name and 'val' in str(t.name).lower() else 1.0
            )
        ))

    fig.update_layout(legend_title_text="Cluster & Split")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Evaluation Metrics Section
    # ---------------------------
    st.markdown("---")
    st.subheader("Cluster Quality Evaluation (METEOR)")
    eval_csv_default = "../embeddings/val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d_cluster_averages.csv"
    eval_csv_path = st.text_input("Path to Evaluation CSV (Cluster Averages)", value=eval_csv_default)

    if os.path.exists(eval_csv_path):
        df_eval = pd.read_csv(eval_csv_path)
        if "cluster" in df_eval.columns and "METEOR" in df_eval.columns:
            df_eval["cluster"] = df_eval["cluster"].astype(str)
            
            # Merge with sample counts if "Train Samples" is available
            eval_merge_cols = ["Train Samples", "Val Samples"] if "Val Samples" in cluster_sizes.columns else ["Train Samples"]
            if set(eval_merge_cols).issubset(cluster_sizes.columns):
                df_merged = pd.merge(cluster_sizes, df_eval, on="cluster", how="inner")
                
                # Filter to match user selections from the sidebar, which allows hiding '-1'
                df_merged = df_merged[df_merged["cluster"].isin(selected_clusters)]
                
                if not df_merged.empty:
                    col_met1, col_met2 = st.columns([1, 1])
                    
                    with col_met1:
                        df_sorted = df_merged.sort_values(by="METEOR", ascending=False)
                        top10 = df_sorted.head(10)
                        bot10 = df_sorted.tail(10).sort_values(by="METEOR", ascending=True)

                        fig_bar = go.Figure()

                        # Top 10 best — green bars
                        fig_bar.add_trace(go.Bar(
                            y=[f"Cluster {c}" for c in top10["cluster"]],
                            x=top10["METEOR"],
                            orientation='h',
                            name="Top 10",
                            marker=dict(
                                color=top10["METEOR"],
                                colorscale="Greens",
                                line=dict(color="darkgreen", width=0.5)
                            ),
                            customdata=top10[["cluster", "Train Samples", "Val Samples"]].values if "Val Samples" in top10.columns else top10[["cluster", "Train Samples"]].values,
                            hovertemplate="<b>Cluster %{customdata[0]}</b><br>METEOR: %{x:.2f}<extra></extra>"
                        ))

                        # Bottom 10 worst — red bars
                        fig_bar.add_trace(go.Bar(
                            y=[f"Cluster {c}" for c in bot10["cluster"]],
                            x=bot10["METEOR"],
                            orientation='h',
                            name="Bottom 10",
                            marker=dict(
                                color=bot10["METEOR"],
                                colorscale="Reds_r",
                                line=dict(color="darkred", width=0.5)
                            ),
                            customdata=bot10[["cluster", "Train Samples", "Val Samples"]].values if "Val Samples" in bot10.columns else bot10[["cluster", "Train Samples"]].values,
                            hovertemplate="<b>Cluster %{customdata[0]}</b><br>METEOR: %{x:.2f}<extra></extra>"
                        ))

                        fig_bar.update_layout(
                            title="Top 10 Best vs Worst Clusters by METEOR",
                            barmode="overlay",
                            xaxis_title="METEOR Score",
                            yaxis_title="Cluster",
                            height=500,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col_met2:
                        # Scatter chart with METEOR background heatmap
                        if "Val Samples" in df_merged.columns:
                            x_col, y_col = "Train Samples", "Val Samples"
                            title_scatter = "Train vs Val Samples — METEOR Heatmap"
                        else:
                            x_col, y_col = "Train Samples", "METEOR"
                            title_scatter = "Train Samples vs METEOR Score"

                        x_vals = df_merged[x_col].values.astype(float)
                        y_vals = df_merged[y_col].values.astype(float)
                        meteor_vals = df_merged["METEOR"].values.astype(float)

                        # Build interpolated background grid (blurred METEOR heatmap)
                        try:
                            from scipy.interpolate import griddata
                            margin_x = (x_vals.max() - x_vals.min()) * 0.1 or 1
                            margin_y = (y_vals.max() - y_vals.min()) * 0.1 or 1
                            xi = np.linspace(x_vals.min() - margin_x, x_vals.max() + margin_x, 100)
                            yi = np.linspace(y_vals.min() - margin_y, y_vals.max() + margin_y, 100)
                            Xi, Yi = np.meshgrid(xi, yi)
                            Zi = griddata((x_vals, y_vals), meteor_vals, (Xi, Yi), method='linear')

                            fig_scatter = go.Figure()
                            fig_scatter.add_trace(go.Heatmap(
                                x=xi, y=yi, z=Zi,
                                colorscale="RdYlGn",
                                zmin=meteor_vals.min(),
                                zmax=meteor_vals.max(),
                                opacity=0.45,
                                showscale=False,
                                hoverinfo='skip'
                            ))
                        except Exception:
                            fig_scatter = go.Figure()

                        # Scatter points on top
                        fig_scatter.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='markers',
                            marker=dict(
                                size=14,
                                color=meteor_vals,
                                colorscale="RdYlGn",
                                showscale=True,
                                colorbar=dict(title="METEOR"),
                                opacity=0.9,
                                line=dict(width=1, color='DarkSlateGrey')
                            ),
                            customdata=df_merged[["cluster", "METEOR"]].values,
                            hovertemplate="<b>Cluster %{customdata[0]}</b><br>METEOR: %{customdata[1]:.2f}<br>" + x_col + ": %{x}<br>" + y_col + ": %{y}<extra></extra>"
                        ))

                        fig_scatter.update_layout(
                            title=title_scatter,
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            height=500
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    # Full sorted bar chart — all clusters
                    st.markdown("##### All Clusters ranked by METEOR")
                    fig_all = px.bar(
                        df_sorted,
                        x="cluster",
                        y="METEOR",
                        title="METEOR Score — All Clusters (Sorted)",
                        color="METEOR",
                        color_continuous_scale="RdYlGn",
                        hover_data=["Train Samples", "Val Samples", "BLEU-1", "ROUGE-L"] if "Val Samples" in df_sorted.columns else ["Train Samples", "BLEU-1", "ROUGE-L"]
                    )
                    fig_all.update_layout(
                        xaxis_type='category',
                        xaxis={'categoryorder': 'total descending'},
                        height=350,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                    )
                    st.plotly_chart(fig_all, use_container_width=True)

                    # Identify worst clusters
                    st.markdown("#### 🚨 Worst Performing Clusters")
                    st.info("The table below highlights the clusters with the worst METEOR scores. Take note of their ID and use the **'Explore images by cluster'** dropdown exactly below this message to inspect their images and discover what makes them so difficult for the model (e.g. blurry photos, heavy text-reliance, empty images).")

                    worst_cols = ["cluster", "METEOR", "Train Samples", "Val Samples", "BLEU-1"] if "Val Samples" in df_merged.columns else ["cluster", "METEOR", "Train Samples", "BLEU-1"]
                    worst_clusters = df_sorted.tail(10)[worst_cols]
                    st.dataframe(worst_clusters, use_container_width=True, hide_index=True)

                else:
                    st.warning("All clusters from the CSV were filtered out (or no match was found).")
        else:
            st.error("The selected CSV does not contain 'cluster' or 'METEOR' columns.")
    else:
        st.info("Evaluation CSV not found. Please provide a valid path to see cluster quality visualization.")
    st.markdown("---")

    # Explore images
    st.subheader("Explore images by cluster")

    col1, col2 = st.columns([1, 1])

    with col1:
        browse_cluster = st.selectbox("Select cluster to browse", all_clusters)
        cluster_rows = df[df["cluster"] == browse_cluster]

        n_per_page = st.slider("Images per page", 5, 50, 20)
        total_pages = max(1, (len(cluster_rows) + n_per_page - 1) // n_per_page)
        page = st.number_input("Page", 1, total_pages, 1) - 1

        start = page * n_per_page
        end = min(start + n_per_page, len(cluster_rows))
        st.write(f"Showing {start + 1}–{end} of {len(cluster_rows)} images in cluster {browse_cluster}")

        cols = st.columns(5)
        for i, (_, row) in enumerate(cluster_rows[start:end].iterrows()):
            base_dir = data_dir if row["split"] == "train" else val_data_dir
            if base_dir is None:
                continue
            img_path = os.path.join(base_dir, row["filename"])
            if os.path.exists(img_path):
                with cols[i % 5]:
                    st.image(img_path, caption=f"{row['filename']} ({row['split']})", width=150)
            else:
                with cols[i % 5]:
                    st.warning(f"Image not found: {img_path}")

    with col2:
        st.markdown("**Search by filename**")
        search = st.text_input("Filename (partial match)")
        if search:
            matches = df[df["filename"].str.contains(search, case=False)]
            st.write(f"Found {len(matches)} matches")
            for _, row in matches.head(10).iterrows():
                base_dir = data_dir if row["split"] == "train" else val_data_dir
                if base_dir is None:
                    continue
                img_path = os.path.join(base_dir, row["filename"])
                if os.path.exists(img_path):
                    st.image(
                        img_path,
                        caption=f"{row['filename']} ({row['split']}, cls {row['cluster']})",
                        width=200,
                    )
                else:
                    st.warning(f"Image not found: {img_path}")

    # Export
    st.sidebar.header("Export")
    
    encoder_name = os.path.basename(embeddings_path).split('_')[0]
    if cluster_method == "HDBSCAN":
        params_str = f"hdbscan_mcs{min_cluster_size}_ms{min_samples}_{cluster_selection}_umap_{umap_metric}_{n_components}d"
    else:
        params_str = f"kmeans_k{n_clusters_km}_umap_{umap_metric}_{n_components}d"
        
    csv_filename_train = f"train_clusters_{encoder_name}_{params_str}.csv"
    csv_filename_val = f"val_clusters_{encoder_name}_{params_str}.csv"
    npz_filename = f"clusters_{encoder_name}_{params_str}.npz"
    
    output_dir = os.path.dirname(embeddings_path) if os.path.dirname(embeddings_path) else "../embeddings"

    # Separate dataframes
    df_train_export = df[df["split"] == "train"]
    df_val_export = df[df["split"] == "val"] if "val" in df["split"].values else None

    if st.sidebar.button("Save results to disk"):
        os.makedirs(output_dir, exist_ok=True)
        save_path_npz = os.path.join(output_dir, npz_filename)
        
        np.savez(
            save_path_npz,
            train_umap=train_umap_result,
            val_umap=val_umap_result if val_umap_result is not None else np.array([]),
            train_labels=train_labels,
            val_labels=val_labels if val_labels is not None else np.array([]),
            train_filenames=filenames,
            val_filenames=val_filenames if val_filenames is not None else np.array([]),
        )
        
        df_train_export.to_csv(os.path.join(output_dir, csv_filename_train), index=False)
        if df_val_export is not None:
            df_val_export.to_csv(os.path.join(output_dir, csv_filename_val), index=False)
            st.sidebar.success(f"Saved Train/Val CSVs and NPZ to {output_dir}")
        else:
            st.sidebar.success(f"Saved Train CSV and NPZ to {output_dir}")

    st.sidebar.download_button(
        "Download Train clusters as CSV",
        df_train_export.to_csv(index=False),
        csv_filename_train,
        "text/csv",
        use_container_width=True,
    )
    
    if df_val_export is not None:
        st.sidebar.download_button(
            "Download Validation clusters as CSV",
            df_val_export.to_csv(index=False),
            csv_filename_val,
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
    parser.add_argument("--data_dir", type=str, default="/ghome/group01/C5/dataset/VizWiz/images/train",
                        help="Path to training images directory")
    parser.add_argument("--val_data_dir", type=str, default="/ghome/group01/C5/dataset/VizWiz/images/val",
                        help="Path to validation images directory")
    parser.add_argument("--embeddings_path", type=str, default="../embeddings/clip_embeddings_train.npz",
                        help="Path to saved train embeddings file")
    parser.add_argument("--val_embeddings_path", type=str, default=None,
                        help="Path to saved validation embeddings file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="../embeddings")
 
    args = parser.parse_args()
 
    if args.step == "extract":
        extract_embeddings(args.data_dir, args.val_data_dir, args.encoder, args.output_dir, args.batch_size)
 
    elif args.step == "visualize":
        run_streamlit(args.embeddings_path, args.val_embeddings_path, args.data_dir, args.val_data_dir)