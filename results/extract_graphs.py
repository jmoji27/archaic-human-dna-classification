import os
import re
import shutil
from datetime import datetime

# Define source directory and target destination folder
SOURCE_DIR = "./cnn_no_pooling"
TARGET_DIR = "./cnn_no_pooling_training_graphs_results_all"

# Regular expression to extract timestamps from filenames (Format: YYYYMMDD_HHMMSS)
TIMESTAMP_REGEX = re.compile(r"(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2})")

def get_timestamp(filename):
    """Extracts a datetime object from the filename if a timestamp exists."""
    match = TIMESTAMP_REGEX.search(filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None

def main():
    # Dictionary to keep track of the latest file for each category per dataset type
    # Structure: {(dataset_type, graph_category): (latest_timestamp, source_file_path, original_filename)}
    latest_graphs = {}

    print(f"Scanning '{SOURCE_DIR}' for training graphs...")

    # Traverse exclusively through the cnn folder structure
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Skip 'test' directories completely to focus solely on training graphs
        if "test" in root.split(os.sep):
            continue

        for file in files:
            # We are only interested in PNG visualizations
            if not file.lower().endswith(".png"):
                continue

            # Determine the dataset folder name from the path context (e.g., bottleneck, multiclass)
            # path components sample: ['./cnn', 'bottleneck', 'graphs']
            relative_parts = os.path.relpath(root, SOURCE_DIR).split(os.sep)
            if not relative_parts or relative_parts[0] == ".":
                continue
            dataset_type = relative_parts[0]

            # Categorize the plot type based on name components
            if "auroc" in file.lower():
                category = "auroc"
            elif "confusion" in file.lower():
                category = "confusion"
            else:
                category = "training_loss_metrics"

            # Parse out the log timestamp
            timestamp = get_timestamp(file)
            if not timestamp:
                continue

            # Check if this file is newer than what we have tracked so far
            key = (dataset_type, category)
            if key not in latest_graphs or timestamp > latest_graphs[key][0]:
                latest_graphs[key] = (timestamp, os.path.join(root, file), file)

    if not latest_graphs:
        print("No matching training graphs found with valid timestamps.")
        return

    # Ensure clean output workspace directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"\nCreated/Verified target directory: {TARGET_DIR}")
    print("Copying most recent graph versions out...")
    print("-" * 70)

    # Copy the selected files over with clean, readable target names
    for (dataset_type, category), (ts, src_path, orig_name) in sorted(latest_graphs.items()):
        # Generate a distinct, clean name so files do not overwrite each other
        timestamp_str = ts.strftime("%Y%m%d_%H%M%S")
        new_filename = f"{dataset_type}_{category}_{timestamp_str}.png"
        dest_path = os.path.join(TARGET_DIR, new_filename)

        shutil.copy2(src_path, dest_path)
        print(f" Found Latest [{category.upper()}] for {dataset_type}:")
        print(f"   → Src:  {src_path}")
        print(f"   → Dest: {dest_path}\n")

    print("-" * 70)
    print(f"Extraction complete! All consolidated files are saved in: {TARGET_DIR}")

if __name__ == "__main__":
    main()
