from PIL import Image
import numpy as np
import torch


def to_min_max_norm_tensor(img):
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    img = (img - img.min()) / (img.max() - img.min())
    return img
    

def to_255_norm_tensor(img):
    img = (torch.tensor(img)/255).unsqueeze(0)
    return img
    


def SaltAndPepper(image, p=.5, sp=.5, amount=.004):
    image = np.array(image)
    if np.random.rand(1)[0] > p:
        salt_coords = (np.random.random(size=(image.shape)) < amount).astype(int)
        image[salt_coords == 1] = 1
        pepper_coords = (np.random.random(size=(image.shape)) < amount).astype(int)
        image[pepper_coords == 1] = 0
        return image
    else:
        return image


def Speckle(image):
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss 
    return noisy
    
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return {
      'images': torch.stack([x[0] for x in batch]).squeeze(1),
      'texts': [x[1] for x in batch]
    }

def collate_fn_255(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return {
      'images': torch.stack([x[0] for x in batch]),
      'texts': [x[1] for x in batch]
    }

def collate_fn_patient_level(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return {
      'oct1': torch.stack([x[0] for x in batch]).squeeze(1),
      'oct2': torch.stack([x[1] for x in batch]).squeeze(1),
      'fundus1': torch.stack([x[2] for x in batch]).squeeze(1),
      'fundus2': torch.stack([x[3] for x in batch]).squeeze(1),
        
      'text': [x[4] for x in batch]
    }


def collate_fn_rnc(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return {
      'images': torch.stack([x[0] for x in batch]).squeeze(1),
      'texts': [x[1] for x in batch],
      'visual_acuity': torch.stack([torch.tensor(x[2]).float() for x in batch])
    }
    