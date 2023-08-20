import os
import numpy as np
import pandas as pd
from utils import mkdir
from tqdm import tqdm
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
        
        
        if no_overlap:
            mask = self.header_df['stride'] == 0
            self.header_df = self.header_df[mask]
        
        self.resize = transforms.Resize((256, 256),
                                       interpolation=transforms.InterpolationMode.BICUBIC,
                                       antialias=True)

        self.lenght = len(self.header_df)
        
        self.make_output_folders()


    def make_output_folders(self):
        vids = self.get_folder_names()
        for vid in tqdm(vids):
            mkdir(f'{self.path}/dyn/{vid}')
    
    def get_folder_names(self):
        return self.header_df['video_id'].unique()

    def __len__(self):
        return self.lenght
    
    def __getitem__(self, index):
        return self.load_clip(index)
    
    def load_clip(self, idx):
        video_id, start, end, stride = self.header_df.iloc[idx][['video_id','start','end','stride']]
        
        frames = []
        for i in range(start, end+1):
            img_path = f'{self.path}/frames/{video_id}/{i}.jpg'
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = np.array(self.resize(image), dtype=np.float32)
            frames.append(image)

        dyn_image = get_dynamic_image(frames, normalized=True)
        
        iio.imsave(f'{self.path}/dyn/{video_id}/{start}.jpg', dyn_image, check_contrast=False)
        return True


def get_dynamic_image(frames, normalized=True):
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""
    num_channels = frames[0].shape[2]
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv.normalize(dynamic_image, None, 0, 255, norm_type=cv.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image

def _get_channel_frames(iter_frames, num_channels):
    """ Takes a list of frames and returns a list of frame lists split by channel. """
    frames = [[] for channel in range(num_channels)]

    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv.split(frame)):
            channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
    for i in range(len(frames)):
        frames[i] = np.array(frames[i])
    return frames

def _compute_dynamic_image(frames):
    """ Adapted from https://github.com/hbilen/dynamic-image-nets """
    num_frames, h, w, depth = frames.shape

    # Compute the coefficients for the frames.
    coefficients = np.zeros(num_frames)
    for n in range(num_frames):
        cumulative_indices = np.array(range(n, num_frames)) + 1
        coefficients[n] = np.sum(((2*cumulative_indices) - num_frames) / cumulative_indices)

    # Multiply by the frames by the coefficients and sum the result.
    x1 = np.expand_dims(frames, axis=0)
    x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
    result = x1 * x2
    return np.sum(result[0], axis=0).squeeze()


    
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
                               num_workers=12, persistent_workers=False, prefetch_factor=10)

    test_dl = data.DataLoader(ds_test, 48, shuffle=False, drop_last=False,
                              num_workers=12, persistent_workers=False, prefetch_factor=10)

    for out in tqdm(train_dl):
        pass

    for out in tqdm(test_dl):
        pass


    print('Obiwan Kenobi!')