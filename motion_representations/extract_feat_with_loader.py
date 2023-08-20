import os
import torch
import numpy as np
import pandas as pd
from utils import mkdir
from tqdm import tqdm, trange

import h5py

from PIL import Image
from torchvision import transforms

from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18, resnet50

class ClipDatasetExtractor():
    
    def __init__(self,
                 root_dir='../data',
                 dataset_name='shanghai',
                 split_type='train',
                 data_type='star',
                 no_overlap=False,
                 size=256,
                ):
        clip_len=16
        self.dataset_name = dataset_name
        self.path = f'{root_dir}/{dataset_name}'
        self.clip_len = clip_len
        self.data_type = data_type
        self.header_df = pd.read_csv(self.path + f'/splits_header_{split_type}.csv')
        
        if no_overlap:
            mask = self.header_df['stride'] == 0
            self.header_df = self.header_df[mask]
        

        self.trans = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                     std=[0.22803, 0.22145, 0.216989]),
                ]
            )


        self.lenght = len(self.header_df)

    
    def get_folder_names(self):
        return self.header_df['video_id'].unique()

    def __len__(self):
        return self.lenght
    
    def __getitem__(self, index):
        return self.load_clip(index)
    
    def load_clip(self, idx):
        video_id, start, end, stride = self.header_df.iloc[idx][['video_id','start','end','stride']]
        
        
        img_path = f'{self.path}/{self.data_type}/{video_id}/{start}.jpg'
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            image = self.trans(image)
        
        return {'data': image, 'vid': video_id ,'idx': start}


# feature extraction
class featureModel(nn.Module):
    def __init__(self, depth=18):
        super(featureModel, self).__init__()
        if depth == 18:
            self.feature = resnet18(pretrained=True)
        else:
            self.feature = resnet50(pretrained=True)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        
    def forward(self, x):
        x = self.feature_forward(x)
        return x
    
    def feature_forward(self, x):
        # x = torch.reshape(x, (-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        x = self.feature.conv1(x)
        x = self.feature.bn1(x)
        x = self.feature.relu(x)
        x = self.feature.maxpool(x)

        x = self.feature.layer1(x)
        x = self.feature.layer2(x)
        x = self.feature.layer3(x)
        x = self.feature.layer4(x)
        # print(x.shape)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # x = torch.reshape(x, (-1, self.clip_len, x.shape[-1]))
        return x


def extract(dataset_name='UCFC',
            data_type='star',
            gpu_id=0,
            depth=18,
            batch_size=48,
            ):
    
    dataset_root: str = '../data'

    feat_model = f'res{depth}'
    feat_size = 512 if depth == 18 else 2048
    
    out_path_root = f'{dataset_root}/{dataset_name}'

    ds = ClipDatasetExtractor(root_dir=dataset_root,
                              dataset_name=dataset_name,
                              split_type='train',
                              data_type=data_type,
                              no_overlap= dataset_name == 'UCFC', 
                              )
    ds_test = ClipDatasetExtractor(root_dir=dataset_root,
                              dataset_name=dataset_name,
                              split_type='test',
                              data_type=data_type,
                              no_overlap=False,
                              )
    
    print('train size', len(ds))
    
    print('test size', len(ds_test))
    
    train_dl = data.DataLoader(ds, batch_size, shuffle=False, drop_last=False,
                               num_workers=6, persistent_workers=False)

    test_dl = data.DataLoader(ds_test, batch_size, shuffle=False, drop_last=False,
                              num_workers=6, persistent_workers=False)

    print('Dataset Change', feat_model)

    model = featureModel(depth=depth).cuda(gpu_id)
    model.eval()

    with torch.no_grad():
        
        with h5py.File(f'{out_path_root}/{data_type}_{feat_model}_feat_train.hdf5', 'w') as f:
            data_h5 = f.create_dataset('data',
                         (len(ds), feat_size),
                        )
            print(data_h5.shape)
            idx = 0
            for batch in tqdm(train_dl):
                image = batch['data']

                out = model(image.cuda(gpu_id)).cpu().numpy()
                for x in out:
                    data_h5[idx] = x[np.newaxis, :]
                    idx += 1


        with h5py.File(f'{out_path_root}/{data_type}_{feat_model}_feat_test.hdf5', 'w') as f:
            data_h5 = f.create_dataset('data',
                         (len(ds_test), feat_size),
                        )
                                
            idx = 0
            for batch in tqdm(test_dl):
                image = batch['data']
                
                out = model(image.cuda(gpu_id)).cpu().numpy()
                for x in out:
                    data_h5[idx] = x[np.newaxis, :]
                    idx += 1

    
  
if __name__ == '__main__':
    
    print('Hello there!')

    dataset_name = 'UCFC'  # UCFC, shanghai
    

    extract(dataset_name=dataset_name,
            data_type='dyn', # star, dyn
            gpu_id=0,
            depth=18,
            batch_size=128,
            )
    
    extract(dataset_name=dataset_name,
            data_type='dyn', # star, dyn
            gpu_id=0,
            depth=50,
            batch_size=128,
            )
    

    print('Obiwan Kenobi!')