import os

import pandas as pd
import shutil

df = pd.read_csv('csv/ukb_data_v2.csv')

id_list = df['Eid'].to_list()
unusable_list = []
data_path = '/mnt/data/ukb_t1_2_0/ukb_t1/'
target_path = '/data/yifan/BrainAgeEstimation/data/t1_data/'
num = 0
num_unusable = 0
for idx in id_list:
    idx = str(idx)
    x = idx + '_20252_2_0'
    source = os.path.join(data_path, x, 'T1', 'T1_brain_to_MNI.nii.gz')
    if not os.path.exists(source):
        print(x, os.listdir(os.path.join(data_path, x, 'T1')))
        num_unusable += 1
        unusable_list.append(int(idx))
    # else:
    #     target = os.path.join(target_path, f'{idx}_20252_2_0.nii.gz')
    #     num += 1
    #     if not os.path.exists(target):
    #         print(f'#{num} copying: ' + x)
    #         # shutil.copy(source, target)

print(num)
print(num_unusable)

pd.DataFrame({'Eid': unusable_list}).to_csv('csv/unusable_id_list.csv', index=False)

