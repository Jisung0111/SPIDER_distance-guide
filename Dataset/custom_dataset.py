import os 
import pandas as pd
import torch 
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import Sampler
import numpy as np
from torchvision.transforms import ToTensor


class Custom2Sampler(Sampler):
    def __init__(self,
                dataset: torch.utils.data.Dataset,
                batch_size: int,
                ):
        super(Custom2Sampler, self).__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size 
        if batch_size % 2 != 0:
            raise ValueError('Batch size should be mutliple of 2') 
        self.num_of_batches = len(self.dataset)//self.batch_size

    def __iter__(self):
        unused_labels = self.dataset.labels
        # # unused_labels = unused_labels.reset_index(level=0, names = 'id')
        #unused_labels['id'] = unused_labels.index
        # unused_labels.to_csv('labels_test.csv', index = False, encoding='utf-8' )
        for _ in range(self.num_of_batches):
            batch = []
            labels = unused_labels.copy() #pandas dataframe
            num_left_slots = self.batch_size #initial number of slots to fill batch
            while num_left_slots > 0:
                random_label = np.random.choice(labels['class'].unique())
                df = labels[labels['class']==random_label].sample(2) # labels is parent data frame, while df is child data frame. 
                for i, rows in df.iterrows():
                    batch.append(rows.index)
                    #changed from rows['ID'] to rows.index
                unused_labels = unused_labels[~unused_labels.loc[df.index]]
                num_left_slots -= 2
            yield np.stack(batch)

    def __len__(self):
        return self.num_of_batches


class CustomDataset(Dataset):
    def __init__(self, labels: pd.DataFrame, root_dir, transform = None):
        '''Arguments:
        label_file: path to csv file which contains 3 columns:
        one with name of the person, other with the label, and last one number of images
        root_dir: path file of images
        transform: transformations that will be applied (default: None)'''
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        '''lenght of the file'''
        return len(self.labels)
    def __getitem__(self, index):
        '''getting image based on index (protocol in MapDatasets)'''
        # print(self.labels.iloc[index, 0], "Here maybe???")
        images = []
        labels = []
        for ind in index: #since our index coming from 
            img_path = os.path.join(self.root_dir, str(self.labels.iloc[ind, 0]))
            image_i = read_image(img_path)
            label_i = torch.tensor(int(self.labels.iloc[ind, 1]))

            if self.transform:
                '''transformations to be done to image'''
                image_i = self.transform(image_i)
            images.append(image_i)
            labels.append(label_i)

        image = np.stack(images)
        label = torch.stack(labels)
        return (image, label)


class ConcateSketchPhoto(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    
    def __getitem__(self, index):
        batch = {}
        for dataset in self.datasets:
            batch = {**batch, **dataset[index]}
        return batch
    def __len__(self):
        return min(len(d) for d in self.datasets)


