from multiprocessing import Pool
from tqdm import trange, tqdm
import pandas as pd
import numpy as np
from time import time
import torch
import os
import pdb
import h5py




class ClipDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root_dir='../data',
                 dataset_name='shanghai',
                 feat_model='r3d18',
                 cond_model='res18',
                 feat_type='data',
                 cond_type='star',
                 split='test',
                 demo=False,
                 no_overlap=False,
                 anomaly=False,
                 no_scale=False,
                 ):

        assert split in ['test', 'train'], 'split type can be train or test'
        assert not(anomaly and demo), 'cannot be both demo and anomaly'

        self.feat_model = feat_model
        self.cond_model = cond_model
        self.feat_type = feat_type
        self.cond_type = cond_type
        self.demo = demo

        self.dataset_name = dataset_name
        self.path = f'{root_dir}/{dataset_name}'
        self.split = split
        
        if split == 'train':
            anomaly = self.dataset_name == 'UCFC'
            no_overlap = self.dataset_name == 'UCFC'
        
        feat_path = f'{self.path}/{self.feat_type}_{self.feat_model}_{split}.hdf5'
        if os.path.exists(feat_path):
            self.f = h5py.File(feat_path, 'r')
            self.feat_data = self.f['data']
        else:
            print('Feature file not found\n',feat_path)

        cond_path = f'{self.path}/{self.cond_type}_{self.cond_model}_{split}.hdf5'
        if os.path.exists(cond_path):
            self.f2 = h5py.File(cond_path, 'r')
            self.cond_data = self.f2['data']
        else:
            print('Condition file not found\n',cond_path)
        

        # labelled header load
        self.header_df = pd.read_csv(self.path + f'/splits_header_{split}.csv')

        if no_overlap:
            indexes = self.header_df['stride'] == 0
            self.header_df = self.header_df[indexes]
                      
        if anomaly:
            indexes = self.header_df['label'] == 1
            self.header_df = self.header_df[indexes]

        if self.demo:
            abnormal_idx = np.where(self.header_df['label']==1)[0]
            normal_idx = np.where(self.header_df['label']==0)[0]
            np.random.shuffle(abnormal_idx)
            np.random.shuffle(normal_idx)
            cut_off = min(len(normal_idx), len(abnormal_idx))
            merge = np.stack([abnormal_idx[:cut_off], normal_idx[:cut_off]], axis=1).flatten()
            self.header_df = self.header_df.loc[merge]

        self.lenght = len(self.header_df)
        if no_scale:

            self.input_scaler = 1.0
            self.cond_scaler = 1.0
        else:
            self.input_scaler = self.decide_scaler(self.feat_type, self.feat_model)
            self.cond_scaler = self.decide_scaler(self.cond_type, self.cond_model)
        
        self.std_data = self.decide_std_data(self.dataset_name, self.feat_type, self.feat_model, no_scale)
        self.std_data2 = self.decide_std_data(self.dataset_name, self.cond_type, self.cond_model, no_scale)


    def decide_scaler(self, feat_type, feat_model):
        if feat_type == 'data':
            scaler = 1.0 if feat_model == 'r3d18' else 2.1
        elif feat_type == 'dyn' or feat_type == 'star_rgb':
            scaler = 0.2 if feat_model == 'res18' else 0.3
        else:
            print('unknown feature type scaler')
            scaler = 1.0
        return scaler

    def decide_std_data(self, dataset_name, feat_type, feat_model, no_scale):
        if no_scale:
            if dataset_name == 'UCFC':
                if feat_type == 'data':
                    std_data = 0.6205 if feat_model == 'r3d18' else 0.3112
                elif feat_type == 'dyn':
                    std_data = 2.0831 if feat_model == 'res18' else 1.3090
                elif feat_type == 'star_rgb':
                    std_data = 2.1074 if feat_model == 'res18' else 1.2460
                else:
                    print('unknown feature type std data')
                    std_data = 1.0
            else:
                if feat_type == 'data':
                    std_data = 0.6205 if feat_model == 'r3d18' else 0.2934
                elif feat_type == 'dyn':
                    std_data = 2.3002 if feat_model == 'res18' else 1.4851
                elif feat_type == 'star_rgb':
                    std_data = 2.5221 if feat_model == 'res18' else 1.3697
                else:
                    print('unknown feature type std data')
                    std_data = 1.0
        else:
            if dataset_name == 'UCFC':
                if feat_type == 'data':
                    std_data = 0.6205 if feat_model == 'r3d18' else 0.6536
                elif feat_type == 'dyn':
                    std_data = 0.4166 if feat_model == 'res18' else 0.3927
                elif feat_type == 'star_rgb':
                    std_data = 0.4215 if feat_model == 'res18' else 0.3738
                else:
                    print('unknown feature type std data')
                    std_data = 1.0
            else:
                if feat_type == 'data':
                    std_data = 0.6206 if feat_model == 'r3d18' else 0.6162
                elif feat_type == 'dyn':
                    std_data = 0.4600 if feat_model == 'res18' else 0.4455
                elif feat_type == 'star_rgb':
                    std_data = 0.5044 if feat_model == 'res18' else 0.4109
                else:
                    print('unknown feature type std data')
                    std_data = 1.0
        return std_data
    
    def __del__(self):
        self.f.close()
        self.f2.close()

    def __len__(self):
        return self.lenght

    def __getitem__(self, index):
        return self.load_clip(index)

    def load_clip(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        
        label = np.array(label, dtype=np.int32)
        clip_feat = self.feat_data[idx] * self.input_scaler
        clip_cond = self.cond_data[idx] * self.cond_scaler

        return {'data': clip_feat, 'cond':clip_cond, 'label':label, 'vid': video_id, 'start': start}


def compute_mean_std(ds):
    from torch.utils import data
    train_dl = data.DataLoader(ds, 512, shuffle=False, drop_last=False,
                               num_workers=6, persistent_workers=False)
    
    def batch_mean_and_sd(loader):
    
        cnt = 0
        fst_moment = torch.zeros(1)
        snd_moment = torch.zeros(1)
        min_value = torch.ones(1) * 9999
        max_value = torch.ones(1) * -9999
        for data_dict in tqdm(loader):
            feat = data_dict['data']
            # print(images.shape)
            b, d = feat.shape
            
            nb_pixels = b * d
            sum_ = torch.sum(feat, dim=[0, 1])
            sum_of_square = torch.sum(feat ** 2, dim=[0, 1])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
            cnt += nb_pixels

            min_value = torch.min(min_value, torch.min(feat))
            max_value = torch.max(max_value, torch.max(feat))

        mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
        return mean.item(), std.item(), min_value.item(), max_value.item()

    mean, std, min_v, max_v = batch_mean_and_sd(train_dl)
    print(f"mean: {mean:.4f}")
    print(f"std : {std:.4f}")
    print(f"max : {max_v:.4f}")
    print(f"min : {min_v:.4f}", )


if __name__ == '__main__':
    
    print('Hello there!')
    from tqdm import tqdm

    dataset_root: str = '../data'
    
    for dataset_name in ['UCFC', 'shanghai']:
        
        feat_type = 'data'
        for f_type in ['r3d18', 'rx3d']:
            ds = ClipDataset(root_dir=dataset_root,
                            dataset_name=dataset_name,
                            feat_model=f_type,
                            cond_model=f_type,
                            feat_type=feat_type,
                            cond_type=feat_type,
                            split='train',
                            no_overlap=True,
                            )
            
            
            print(dataset_name)
            print(len(ds))
            print(ds.std_data, ds.std_data2)
            print(feat_type, f_type)
            compute_mean_std(ds)

        for feat_type in ['dyn', 'star_rgb']:
            for f_type in ['res18', 'res50']:
            
                ds = ClipDataset(root_dir=dataset_root,
                        dataset_name=dataset_name,
                        feat_model=f_type,
                        cond_model=f_type,
                        feat_type=feat_type,
                        cond_type=feat_type,
                        split='train',
                        no_overlap=True,
                        )
                
                print(dataset_name)
                print(len(ds))
                print(ds.std_data, ds.std_data2)
                print(feat_type, f_type)
                compute_mean_std(ds)
        
        
