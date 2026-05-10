import os
import pandas as pd
import gc

DATASETS = {
    "original":               ("dataset/binary/original",               {0: "Human",     1: "Denisovan"}),
    "longerbp":               ("dataset/binary/longerbp",               {0: "Human",     1: "Denisovan"}),
    "bottleneck":             ("dataset/binary/bottleneck",             {0: "Human",     1: "Denisovan"}),
    "multiclass":             ("dataset/multiclass/original",                    {0: "Human",     1: "Denisovan", 2: "Neanderthal"}),
    "HumanvsNeanderthal":     ("dataset/binary/HumanvsNeanderthal",     {0: "Human",     1: "Neanderthal"}),
    "DenisovanvsNeanderthal": ("dataset/binary/DenisovanvsNeanderthal", {0: "Denisovan", 1: "Neanderthal"}),
}


def analyze_split(csv_path, class_map):
    # only load the two columns you need, nothing else
    df = pd.read_csv(csv_path, low_memory=False, usecols=["sequence", "label"],
                     dtype={"sequence": "str", "label": "int32"})

    lengths = df["sequence"].str.len()
    labels  = df["label"]
    total   = len(df)

    print(f"    Total samples : {total:,}")
    print(f"    Seq length — min: {lengths.min()} | max: {lengths.max()} | mean: {lengths.mean():.1f} | median: {lengths.median():.0f}")

    for class_id, class_name in class_map.items():
        count = (labels == class_id).sum()
        pct   = count / total * 100
        print(f"    Class {class_id} ({class_name:>12}) : {count:>8,}  ({pct:.1f}%)")

    # free memory immediately after each split
    del df, lengths, labels
    gc.collect()


def main():
    for dataset_name, (base_path, class_map) in DATASETS.items():
        print(f"\n{'='*65}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*65}")

        for split in ["train", "val", "test"]:
            csv_path = os.path.join(base_path, f"{split}.csv")
            if not os.path.exists(csv_path):
                print(f"  [{split}] — NOT FOUND at {csv_path}")
                continue

            print(f"  [{split}]")
            analyze_split(csv_path, class_map)


if __name__ == "__main__":
    main()