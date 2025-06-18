from torch.utils.data import Dataset
import torch
from util_funcs import image_loading, image_loading_255, image_no_norm
from random import sample


class ImageCaptionDataset(Dataset):
    def __init__(self, dataset_csv, preprocess=None, load_255=False):
        self.dataset_csv = dataset_csv
        self.oct_dict = {'left': 'OS_OCT',
             'right': 'OD_OCT'}
        self.fundus_dict = {'left': 'OS_FUNDUS',
            'right': 'OD_FUNDUS'}
        self.preprocess = preprocess
        self.load_255 = load_255
    
    def __len__(self):
        return len(self.dataset_csv.scan)
        
    def __getitem__(self, idx):
        scan = self.dataset_csv.scan[idx]    
        eye = self.dataset_csv.eye[idx]

        oct_text = self.dataset_csv[self.oct_dict[eye]][idx]
                                            
        if self.preprocess is not None:
            if self.load_255:
                image = image_loading_255(scan)
            else:
                image = image_loading(scan)
            image = self.preprocess(image)
        else:
            image = image_loading(scan)
            
        return [image, oct_text]

