#!/bin/bash

DATA_DIR="./dataset/VizWiz"

echo "Creating directory $DATA_DIR"
mkdir -p "$DATA_DIR/images"
mkdir -p "$DATA_DIR/annotations"

echo "Downloading annotations..."
wget -c "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip" -P "$DATA_DIR/annotations/"
echo "Unzipping annotations..."
unzip -q -n "$DATA_DIR/annotations/annotations.zip" -d "$DATA_DIR/annotations/"

echo "Downloading validation images..."
wget -c "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip" -P "$DATA_DIR/images/"
echo "Unzipping validation images..."
unzip -q -n "$DATA_DIR/images/val.zip" -d "$DATA_DIR/images/"

echo "Downloading test images..."
wget -c "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip" -P "$DATA_DIR/images/"
echo "Unzipping test images..."
unzip -q -n "$DATA_DIR/images/test.zip" -d "$DATA_DIR/images/"

echo "Downloading train images..."
wget -c "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip" -P "$DATA_DIR/images/"
echo "Unzipping train images..."
unzip -q -n "$DATA_DIR/images/train.zip" -d "$DATA_DIR/images/"

echo "Download and extraction complete!"
