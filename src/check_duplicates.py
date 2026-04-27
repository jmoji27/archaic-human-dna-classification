# check_duplicates.py
import pandas as pd

def quick_duplicate_check(train_path, val_path):
    """Quick check for duplicate issues."""
    
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    
    print("="*60)
    print("DUPLICATE ANALYSIS")
    print("="*60)
    
    # Cross-class duplicates in training set
    train_0 = set(train[train['label'] == 0]['sequence'])
    train_1 = set(train[train['label'] == 1]['sequence'])
    cross_class = train_0 & train_1
    
    print(f"\nTRAIN SET:")
    print(f"  Total samples: {len(train):,}")
    print(f"  Class 0 unique sequences: {len(train_0):,}")
    print(f"  Class 1 unique sequences: {len(train_1):,}")
    print(f"  🚨 CROSS-CLASS DUPLICATES: {len(cross_class):,}")
    print(f"     ({100*len(cross_class)/len(train):.2f}% of dataset)")
    
    # Data leakage: train sequences in val
    val_seqs = set(val['sequence'])
    leakage = train_0.union(train_1) & val_seqs
    
    print(f"\nDATA LEAKAGE:")
    print(f"  Train sequences appearing in val: {len(leakage):,}")
    print(f"     ({100*len(leakage)/len(val):.2f}% of val set)")
    
    if len(cross_class) > 0:
        print("\n⚠️  WARNING: You have cross-class duplicates!")
        print("   This means the SAME sequence has DIFFERENT labels.")
        print("   The model cannot learn this - it's pure noise.")
        print("   → Run the cleaning pipeline ASAP!")
    
    print("="*60)

# Run it
quick_duplicate_check(
    'dataset/binary/original/train.csv',
    'dataset/binary/original/val.csv'
)