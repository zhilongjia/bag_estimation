import pandas as pd
import numpy as np
import random

df = pd.read_csv('csv/ukb_data_v2.csv')

df_heal = df[df['is_healthy'] == 1].copy()
seed = 12345

train = np.ones([len(df_heal)], dtype=np.int32)
sample_size = int(len(df_heal) * 0.1 + 1)

# 10% of the healthy people for testing
random.seed(seed)
sampled_idx = random.sample(range(len(df_heal)), sample_size)

train[sampled_idx] = 0

df_heal['train'] = train
print(df_heal['train'].value_counts())

df_heal.to_csv('csv/ukb_data_healthy_v2.csv', index=False)
