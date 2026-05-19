import os
import json
import pandas as pd 

from sklearn.metrics import classification_report

CLASS_NAMES = {
    "original":               ["Human", "Denisovan"],
    "longerbp":               ["Human", "Denisovan"],
    "bottleneck":             ["Human", "Denisovan"],
    "multiclass":             ["Human", "Denisovan", "Neanderthal"],
    "HumanvsNeanderthal":     ["Human", "Neanderthal"],
    "DenisovanvsNeanderthal": ["Denisovan", "Neanderthal"],
}

def save_to_results_csv(all_labels, all_preds, labels_names, accuracy, mcc, model_name, dataset_type):
    report_dict = classification_report(all_labels, all_preds, target_names=labels_names, digits=4,  output_dict=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []

    for class_name in labels_names:
        rows.append({
            "timestamp": timestamp,
            "model": model_name,
            "dataset": dataset_type,
            "class": class_name,
            "precision": round(report_dict[class_name]["precision"], 4),
            "recall": round(report_dict[class_name]["recall"], 4),
            "f1-score": round(report_dict[class_name]["f1-score"], 4),
            "support": report_dict[class_name]["support"],
            "accuracy": round(accuracy, 4),
            "mcc": round(mcc, 4)
        })

    path = "results/results.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_new = pd.DataFrame(rows)

    if os.path.exists(path):
        df_existing = pd.read_csv(path)

        already_exists = (
            (df_existing["model"] == model_name)&
            (df_existing["dataset"] == dataset_type)
        ).any()

        if already_exists:
            print(f"Warning: Results for model '{model_name}' on dataset '{dataset_type}' already exist in results.csv. Skipping save to avoid duplicates.")
            return
        
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(path, index=False)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
