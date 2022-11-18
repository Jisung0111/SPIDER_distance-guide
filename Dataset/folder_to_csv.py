import pandas as pd
import os
import numpy as np

def folder_pandas(root_path):
    label = {}
    df = {}
    for file in os.listdir(root_path):
        if file[:-13] not in label.keys():
            if label:
                index = max(label.values()) + 1
            else:
                index = 0
            label.update({file[:-13]: index})
            df.update({file: index})
        else: 
            index = label[file[:-13]]
            df.update({file:index})
    dataframe = pd.DataFrame(list(df.items()), columns = ['file path', 'class'])
    return dataframe

df = folder_pandas('/home/siraj/sketches/Face-Sketch-Wild/vggface_10')
# df['id'] = df.index
df.to_csv('labels.csv', index = False, encoding='utf-8' )

