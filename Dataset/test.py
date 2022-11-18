from custom_dataset import *
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import argparse


parser.add_argument('--root_dir_photos', type=str, required=True)
parser.add_argument('--root_dir_skethces', type=str, required=True)
# Parse the argument
args = parser.parse_args()

if __name__ == '__main__':
    '''There is a problem with dataframe division, i.e when 
    splitting to training, val, and test, dataframe suprisingly
    uses indexes of previous values. 
    '''
    def dataset_split(label_file: str, num_classes: int, p: list) -> pd.DataFrame: 
        '''This function divides label file such that 
        samples of same person will be appear only either train, val, or test
        '''
        df = pd.read_csv(label_file)
        num_train = int(num_classes*p[0])
        num_val = int(num_classes*p[1])
        num_classes_array = np.arange(num_classes)
        train_classes = np.random.choice(num_classes_array, num_train, replace = False)
        val_classes = np.random.choice(num_classes_array, num_val, replace=False)
        test_classes = num_classes_array
        df_train = df[df['class'].isin(train_classes)]
        df_val = df[df['class'].isin(val_classes)]
        df_test = df[df['class'].isin(test_classes)]
        print(len(df_train), len(df_val), len(df_test), "LENGTHS")
        return df_train, df_val, df_test

        
    df_train, df_val, df_test = dataset_split('labels.csv', 2000, p = [0.8, 0.1])
    datasets = [] #this is for dataset concatenation
    

    train_dataset_photos= CustomDataset(labels = df_train, root_dir = arg.root_dir_photos)
    datasets.append(train_dataset_photos)
    
    # val_dataset_photos= CustomDataset(labels = df_val, root_dir =  arg.root_dir_photos, transform=ToTensor)
    # test_dataset_photos= CustomDataset(labels = df_test, root_dir =  arg.root_dir_photos, transform=ToTensor)
    custom_sampler_train = Custom2Sampler(train_dataset_photos, batch_size = 10)
    # custom_sampler_val = Custom2Sampler(val_dataset_photos, batch_size = 10)
    # custom_sampler_test = Custom2Sampler(test_dataset_photos, batch_size = 10)


    
    train_dataset_sketches= CustomDataset(labels = df_train, root_dir = arg.root_dir_sketches)
    datasets.append(train_dataset_sketches)
    # val_dataset_sketches= CustomDataset(labels = df_val, root_dir = arg.root_dir_sketches, transform=ToTensor)
    # test_dataset_sketches= CustomDataset(labels = df_test, root_dir = arg.root_dir_sketches', transform=ToTensor)

    datald= DataLoader(dataset=train_dataset_photos, sampler = custom_sampler_train)
    features, labels = next(iter(datald))
    print(features.shape, labels.shape)

    '''If you want to test whole dataset division, 
    uncomment below lines to test
    Note: So far only this works'''
    # photos_dataframe = pd.read_csv('labels.csv')
    # photos_dataset = CustomDataset(labels=photos_dataframe, root_dir =  arg.root_dir_photos)
    # photos_sampler = Custom2Sampler(photos_dataset, batch_size=10)
    # dataloader_photos = DataLoader(dataset = photos_dataset, sampler = photos_sampler)
    # features, labels = next(iter(datald))
    # print(features.shape, labels.shape, labels)
    
