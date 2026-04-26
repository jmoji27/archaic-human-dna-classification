#!/bin/bash

# No spaces around the = sign!
TARGET_DIR="$HOME/archaic-human-dna-classification/data/human"

# Move into the directory so the rename stays local
cd "$TARGET_DIR" || { echo "Directory not found"; exit 1; }

for f in Homo_sapiens.GRCh38.dna.chromosome.*.fa; do
    # Check if the file exists (prevents errors if no files match)
    [ -e "$f" ] || continue
    
    # Extract the number
    num=$(echo "$f" | sed 's/.*chromosome\.\([0-9]*\)\.fa/\1/')
    
    echo "Renaming $f to human_chr$num.fa"
    mv "$f" "human_chr$num.fa"
done
