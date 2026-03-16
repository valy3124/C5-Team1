#!/bin/bash

# 1. Setup Directory Structure
echo "Creating folder structure..."
mkdir -p DEArt/images
mkdir -p DEArt/annotations

# 2. Download and Extract Annotations
echo "Downloading Annotations..."
wget -q --show-progress -O DEArt/annotations/annots.zip https://zenodo.org/record/6984525/files/annots_pub.zip

echo "Extracting Annotations..."
unzip -q DEArt/annotations/annots.zip -d DEArt/annotations/

echo "Annotations ready."
rm DEArt/annotations/annots.zip  # Remove zip

# 3. Download and Extract Images (Loop 1 to 16)
echo "Downloading and Extracting Images (this may take time)..."

for i in {01..16}; do
    url="https://zenodo.org/record/6984525/files/images_pub_$i.zip"
    zip_file="DEArt/images/part_$i.zip"
    
    echo "Processing Part $i/16..."
    
    # Download
    wget -q --show-progress -O "$zip_file" "$url"
    
    # Extract directly into DEArt/images/ (then remove zip)
    unzip -q "$zip_file" -d DEArt/images/
    rm "$zip_file"
done

echo "------------------------------------------------"
echo "Download Complete!"
echo "Images are in: DEArt/images/"
echo "Annotations are in: DEArt/annotations/"
echo "All zip files have been removed."