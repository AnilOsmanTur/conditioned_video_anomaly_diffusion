import os
import torch
import numpy as np
import pandas as pd
import shutil
from utils import mkdir
from tqdm import tqdm, trange
import cv2 as cv
from skimage import io as iio

from PIL import Image
from torchvision import transforms



class ClipDatasetExtractor():
    
    def __init__(self,
                 root_dir='../data',
                 dataset_name='shanghai',
                 split_type='train',
                 no_overlap=False,
                ):
        clip_len=16
        self.dataset_name = dataset_name
        self.path = f'{root_dir}/{dataset_name}'
        self.clip_len = clip_len
        
        self.header_df = pd.read_csv(self.path + f'/splits_header_{split_type}.csv')
        
        self.stat_dict = {}
        self.get_stats()

        if no_overlap:
            mask = self.header_df['stride'] == 0
            self.header_df = self.header_df[mask]
        
        self.resize = transforms.Resize((256, 256),
                                       interpolation=transforms.InterpolationMode.BICUBIC,
                                       antialias=True)

        self.jump_ch = ((self.clip_len-1) // 3)
        self.lenght = len(self.header_df)

    def get_stats(self):
        header = pd.read_csv(self.path + '/header.csv')
        # header = pd.read_csv(path + '/header.csv')
        vid_fc = header[['video_id','frame_count']].values
        for vid, fc in tqdm(vid_fc):
            vid_star = np.load(f'{self.path}/star/{vid}.npy')
            star_min = np.min(vid_star)
            star_max = np.max(vid_star)
            star_mean = np.mean(vid_star)
            star_std = np.std(vid_star)
            self.stat_dict[vid]=[fc, star_min, star_max, star_mean, star_std]
            mkdir(f'{self.path}/star/{vid}')
            mkdir(f'{self.path}/star_rgb/{vid}')
    
    def get_folder_names(self):
        return self.header_df['video_id'].unique()

    def __len__(self):
        return self.lenght
    
    def __getitem__(self, index):
        return self.load_clip(index)
    
    def load_clip(self, idx):
        video_id, start, end, stride = self.header_df.iloc[idx][['video_id','start','end','stride']]
        fc, star_min, star_max, star_mean, star_std = self.stat_dict[video_id]
        
        count = 0
        star_id = 0
        img_path = f'{self.path}/frames/{video_id}/{start}.jpg'
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            prev = np.array(self.resize(image), dtype=np.float32)
        
        star_2 = np.zeros(shape=(prev.shape[-3:-1]),dtype=np.float64)
        star_rgb_2 = np.zeros(shape=(3, *prev.shape[-3:-1]),dtype=np.float64)
        for i in range(start+1, end+1):
            img_path = f'{self.path}/frames/{video_id}/{i}.jpg'
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                cur = np.array(self.resize(image), dtype=np.float32)

            x = np.matmul(cur[:,:,np.newaxis,:], prev[:,:,np.newaxis,:], axes=[(-2,-1),(-1,-2),(-2,-1)])
            x = x.squeeze()
            cosine = x / (np.linalg.norm(cur, 2, axis=-1)*np.linalg.norm(prev, 2, axis=-1) + 0.000001)
            alpha = 1 - cosine
            
            clip_dif = np.linalg.norm(prev, 2, axis=-1) - np.linalg.norm(cur, 2, axis=-1)
            clip_dif = np.abs(clip_dif)
            
            star_2 += (1 - alpha / 2) * clip_dif * 1 / fc
            star_rgb_2[star_id] += (1 - alpha / 2) * clip_dif * 1 / fc
            
            prev = cur
            count += 1
            if count % self.jump_ch == 0:
                count = 0
                star_id += 1

        star_2 = (star_2 - star_mean) / star_std
        star_rgb_2 = (star_rgb_2 - star_mean) / star_std
        star_rgb_2 = star_rgb_2.transpose(1,2,0)
        
        star_2 = cv.normalize(star_2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        star_rgb_2 = cv.normalize(star_rgb_2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        iio.imsave(f'{self.path}/star/{video_id}/{start}.jpg', star_2, check_contrast=False)
        iio.imsave(f'{self.path}/star_rgb/{video_id}/{start}.jpg', star_rgb_2, check_contrast=False)
        # iio.imsave(f'out.jpg', star_2, check_contrast=False)
        # iio.imsave(f'out2.jpg', star_rgb_2, check_contrast=False)
        return True


def cal_star_reps_mat(clip):
    prev = clip[:-1].astype(np.float32)
    next = clip[1:].astype(np.float32)

    x = np.matmul(next[:,:,:,np.newaxis,:], prev[:,:,:,np.newaxis,:], axes=[(-2,-1),(-1,-2),(-2,-1)])
    x = x.squeeze()
    cosine = x/(np.linalg.norm(next, 2, axis=-1)*np.linalg.norm(prev, 2, axis=-1) + 0.000001)
    alpha = 1 - cosine

    clip_dif = np.linalg.norm(clip[1:], 2, axis=-1) - np.linalg.norm(clip[:-1], 2, axis=-1)
    clip_dif = np.abs(clip_dif)
    clip_dif = (1 - alpha / 2) * clip_dif

    star_2 = np.sum(clip_dif, axis=0) / len(clip)
    star_2[star_2 < 2*np.mean(star_2)] = 0
    
    w,h = clip_dif.shape[-2:]
    star_rgb_2 = np.reshape(clip_dif, (3,-1,w,h))

    star_rgb_2 = np.sum(star_rgb_2, axis=1) / len(clip)
    star_rgb_2 = star_rgb_2.transpose(1,2,0)
    star_rgb_2[star_rgb_2 < 2*np.mean(star_rgb_2)] = 0
    
    return star_rgb_2, star_2

    
if __name__ == '__main__':
    
    print('Hello there!')
    from tqdm import tqdm

    dataset_root: str = '../data'
    dataset_name = 'UCFC'  # UCFC, shanghai

    ds = ClipDatasetExtractor(root_dir=dataset_root,
                              dataset_name=dataset_name,
                              split_type='train',
                              no_overlap= dataset_name == 'UCFC', 
                              )
    ds_test = ClipDatasetExtractor(root_dir=dataset_root,
                              dataset_name=dataset_name,
                              split_type='test',
                              no_overlap=False,
                              )
    
    print('train size', len(ds))
    
    print('test size', len(ds_test))

    from torch.utils import data
    train_dl = data.DataLoader(ds, 48, shuffle=False, drop_last=False,
                               num_workers=12, persistent_workers=False)

    test_dl = data.DataLoader(ds_test, 48, shuffle=False, drop_last=False,
                              num_workers=12, persistent_workers=False)

    for out in tqdm(train_dl):
        pass

    for out in tqdm(test_dl):
        pass


    print('Obiwan Kenobi!')