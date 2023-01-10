import json
from pathlib import Path
from typing import List
import os

import torch
import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from transformers import CLIPTokenizer
base_path = Path(__file__).absolute().parents[1].absolute()

def preprocess(image,dim=None):
    if dim :
        image = image.resize((dim,dim), resample=PIL.Image.LANCZOS)
    else :
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = np.array(image).astype(np.float32) / 255.0
    if len(image.shape)<3:
        image=np.stack((image,)*3,-1)
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0



def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)






class FashionIQDataset(Dataset):
    """
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str],tokenizer:CLIPTokenizer,display=False,dim=512):

        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")
        self.split=split
        self.dim=dim
        self.display=display
        self.tokenizer=tokenizer
        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(base_path / 'data' / 'captions' / f'cap.{dress_type}.{split}1.json') as f:
                self.triplets.extend(json.load(f))
        #self.triplets=self.triplets# mode
        # get the image names
        # self.image_names: list = []
        # for dress_type in dress_types:
        #     with open(base_path / 'data' / 'image_splits' / f'split.{dress_type}.{split}1.json') as f:
        #         self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset initialized")

    def __getitem__(self, index):
        try:
            image_captions = self.triplets[index]['captions']
            example={}
            example['cap_word']=str(image_captions[0]+', '+image_captions[1])

            if not self.display:
                image_captions=self.tokenizer(
                    example['cap_word'],
                    padding="max_length",
                    truncation=True,
                    #max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
                example['caption']=image_captions
                
                
            reference_name = self.triplets[index]['candidate']
            example['ref_name']=reference_name
            if self.split == 'train'or self.split=='val':
                reference_image_path = base_path / 'data' / 'images' / f"{reference_name}.jpg"
                reference_image = preprocess(PIL.Image.open(reference_image_path),self.dim)
                reference_image_dim =preprocess(PIL.Image.open(reference_image_path),224)
                target_name = self.triplets[index]['target']
                target_image_path = base_path / 'data' / 'images' / f"{target_name}.jpg"
                target_image =preprocess(PIL.Image.open(target_image_path),self.dim)
                example['ref']=reference_image.squeeze(0)
                example['ref_i']=reference_image_dim.squeeze(0)
                example['target']=target_image.squeeze(0)
                if self.display:
                    example['file_name']=example['cap_word'].replace(' ','_').replace('/','?')
                    example['target_name']=target_name
                    example['ref_name']=reference_name


            elif self.split == 'test':
                reference_image_path = base_path / 'data' / 'images' / f"{reference_name}.jpg"
                reference_image = preprocess(PIL.Image.open(reference_image_path),self.dim)
                example['ref']=reference_image.squeeze(0)
                example['target']=None
                example['caption']=image_captions
            return example
        except Exception as e:
           # print(f"ref img shape{reference_image.shape}, tar img shape {target_image_path}")
            # print(f'target name {target_name}, ref name{reference_name}')
            # print(example)
            print(f"Exception: {e}")

    def __len__(self):
        return len(self.triplets)
    
    
    
class Fashion200kDataset(Dataset):
    def __init__(self,split:str,dress_type:List[str],data_path:str,tokenizer:CLIPTokenizer,dim=None) -> None:
        self.split=split
        self.dress_type=dress_type
        self.dim=dim
        self.data_path=data_path
        self.pairs=[]
        self.tokenizer=tokenizer
        for i in dress_type:
            with open(f'{data_path}/labels/{i}_{split}_detect_all.txt','r') as f:
                for line in f.readlines():
                    pair={}
                    pair['src']=line.split('\t')[0].replace('women',data_path)
                    pair['caption']=line.split('\t')[-1].strip('\n')
                    self.pairs.append(pair)
                
            f.close()
    
    def __getitem__(self, index):
        image_caption=self.pairs[index]['caption']
        image=preprocess(PIL.Image.open(self.pairs[index]['src']),self.dim)
        sample={}
        sample['caption']=self.tokenizer(
                    image_caption,
                    padding="max_length",
                    truncation=True,
                    #max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
        sample['target']=image.squeeze(0)
        return sample
                
                
    def __len__(self):
        return len(self.pairs)
            
if __name__ =='__main__':
    p=Fashion200kDataset('train',['dress','jacket','skirt','top','pants'],'fashion200k',dim=512)