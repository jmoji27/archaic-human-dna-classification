
import pandas as pd
df = pd.read_csv('dataset/multiclass/original/train.csv')
print(df['label'].value_counts(normalize=True))

