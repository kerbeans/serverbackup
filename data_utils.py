import json
from pathlib import Path
from typing import List
import os
from tqdm import tqdm
import torch
import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from transformers import CLIPTokenizer,CLIPFeatureExtractor, CLIPVisionModel,CLIPTextModel
from torchvision import transforms
base_path = Path(__file__).absolute().parents[1].absolute()
IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

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

def choose_best(file_list,folder,diffusion_path='model',device='cuda'):
    vision_model=CLIPVisionModel.from_pretrained(f'{diffusion_path}/clip-vit-base-patch32').to(device)
    text_encoder=CLIPTextModel.from_pretrained(f'{diffusion_path}', subfolder="text_encoder").to(device)
    text_tokenizer=CLIPTokenizer.from_pretrained(f'{diffusion_path}', subfolder="tokenizer")
    img_preprocessor=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
    triplets=[]
    res=[]
    for dress_type in ['dress','shirt','toptee']:
        with open(base_path / 'data' / 'captions' / f'cap.{dress_type}.val1.json') as f:
            triplets.extend(json.load(f))
    for i in tqdm(file_list):
        ref_img_name=i[0].split("_")[0]
        tar_img_name=i[0].split('_')[-1]
        caption=[f"{tr['captions'][0]}. {tr['captions'][1]}" for tr in triplets 
                 if (tr['candidate']== ref_img_name) and (tr['target']==tar_img_name)]
        #print(i)#
        caption=text_tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
                ).input_ids[0].to(device)
        #print(caption.shape,'ddddd') [77]
        t_m=text_encoder(caption.unsqueeze(0))[1]
        #print(f'text shape {t_m.shape}')# 1,768
        pixel_value_gen=torch.stack([torch.from_numpy(img_preprocessor(PIL.Image.open(f'{folder}/{j}.jpg'))['pixel_values'][0]).to(device) for j in i],dim=0)
        i_m_gen=vision_model(pixel_values=pixel_value_gen)[1]# b* f
        pixel_value_ref=torch.from_numpy(img_preprocessor(PIL.Image.open(f'data/images/{ref_img_name}.jpg'))['pixel_values'][0]).to(device).unsqueeze(0)
        i_m_ref=vision_model(pixel_values=pixel_value_ref)[1]
        compose=i_m_ref+t_m# 1*f
        table= i_m_gen@compose.T
        table=table.detach().to('cpu').numpy()
        #print(table.shape,table.max(0),np.argmax(table,0))#
        res.append(i[np.argmax(table,0)[0]]) 
        #print('res is ',res)
    print('choosen finished')   
    return res


class FashionIQDataset(Dataset):
    """
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str],tokenizer:CLIPTokenizer=None,preprossor:CLIPFeatureExtractor=None,display=False,dim=512,light=False):

        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")
        self.split=split
        self.light=light
        self.dim=dim
        self.display=display
        self.tokenizer=tokenizer
        self.img_preprocessor=preprossor
        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(base_path / 'data' / 'captions' / f'cap.{dress_type}.{split}1.json') as f:
                self.triplets.extend(json.load(f))
        if light:
            self.img_transformer_for_evaluation=transforms.Compose([transforms.Resize((self.dim, self.dim)), transforms.ToTensor(),
                               transforms.Normalize(**IMAGENET_STATS)])
            can=set([i['candidate'] for i in self.triplets])
            tar=set([i['target'] for i in self.triplets])
            self.val_unique=list(can|tar)
        #self.triplets=self.triplets# mode
        # get the image names
        # self.image_names: list = []
        # for dress_type in dress_types:
        #     with open(base_path / 'data' / 'image_splits' / f'split.{dress_type}.{split}1.json') as f:
        #         self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset initialized size {len(self.triplets)}" )

    def __getitem__(self, index):
        if self.light and self.split=='val':
            sample={}
            sample['img_name']=self.val_unique[index]
            img_path= base_path / 'data' / 'images' / f"{sample['img_name']}.jpg"
            if self.img_preprocessor is None :
                sample['img_input']=self.img_transformer_for_evaluation(PIL.Image.open(img_path).convert("RGB"))
            else :
                sample['img_input']=self.img_preprocessor(PIL.Image.open(img_path).convert("RGB"))['pixel_values'][0]
            
            return sample
        try:
            image_captions = self.triplets[index]['captions']
            example={}
            example['cap_word']=str(image_captions[0]+', '+image_captions[1])

            if not self.display:
                image_captions=self.tokenizer(
                    example['cap_word'],
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
                example['caption']=image_captions
                
                
            reference_name = self.triplets[index]['candidate']
            example['ref_name']=reference_name
            if self.split == 'train'or self.split=='val':
                reference_image_path = base_path / 'data' / 'images' / f"{reference_name}.jpg"
                reference_image = preprocess(PIL.Image.open(reference_image_path),self.dim)
                reference_image_dim =self.img_preprocessor(PIL.Image.open(reference_image_path))
                target_name = self.triplets[index]['target']
                target_image_path = base_path / 'data' / 'images' / f"{target_name}.jpg"
                target_image =preprocess(PIL.Image.open(target_image_path),self.dim)
                example['ref']=reference_image.squeeze(0)## modifed switch to tensor
                example['ref_i']=reference_image_dim['pixel_values'][0]
                example['tar_i']=self.img_preprocessor(PIL.Image.open(target_image_path))['pixel_values'][0]
                example['target']=target_image.squeeze(0)
                if True:
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
    


class FashionIQsmall(Dataset):
    def __init__(self,file_path,dim=224,multi=0):
        self.dim=dim
        self.file_path=file_path
        self.imgs=[]
        self.multi=multi
        for root,dir,files in os.walk(file_path):
            for file in files:
                if 'jpg' in file:
                    self.imgs.append(file.replace('.jpg',""))
        # self.imgs=[i for i in self.imgs if i.split('_')[-2]=='1']
        if multi>= 1:
            type_list=set([f"{i.split('_')[0]}_{i.split('_')[-1]}" for i in self.imgs])  
            imgs_type=[]
            for i in type_list:  
                imgs_type.append([k for k in self.imgs if 
                          (k.split('_')[0]==i.split('_')[0])and(k.split('_')[-1]==i.split("_")[-1])])
            self.imgs=imgs_type
            if multi==2:
                self.imgs=choose_best(self.imgs,file_path,'model/info_nce_dc_128_1e-06_40','cuda:3')
        self.img_transformer_for_evaluation=transforms.Compose([transforms.Resize((self.dim, self.dim)), transforms.ToTensor(),
            transforms.Normalize(**IMAGENET_STATS)])
        if multi ==-1:
            self.img_transformer_for_evaluation=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
        
    def __getitem__(self, index) -> dict:
        if self.multi==0 or self.multi ==2:
            sample={}
            sample['ref_name']=self.imgs[index].split("_")[0]
            sample['tar_name']=self.imgs[index].replace(".jpg","").split("_")[-1]
            sample['gen_input']=self.img_transformer_for_evaluation(PIL.Image.open(f"{self.file_path}/{self.imgs[index]}.jpg"))
            return sample
        elif self.multi==1:
            sample={}
            sample['ref_name']=self.imgs[index][0].split("_")[0]
            sample['tar_name']=self.imgs[index][0].replace(".jpg","").split("_")[-1]
            sample['gen_input']=[]
            for i in self.imgs[index]:
                sample['gen_input'].append(self.img_transformer_for_evaluation(
                    PIL.Image.open(f"{self.file_path}/{i}.jpg")))
            sample['gen_input']=torch.stack(sample['gen_input'],dim=0)
            return sample
        elif self.multi ==-1:
            sample={}
            sample['ref_name']=self.imgs[index].split("_")[0]
            sample['tar_name']=self.imgs[index].replace(".jpg","").split("_")[-1]
            sample['gen_input']=self.img_transformer_for_evaluation(
                PIL.Image.open(f"{self.file_path}/{self.imgs[index]}.jpg").convert('RGB'))['pixel_values'][0]
            return sample
    def __len__(self):
        return len(self.imgs)
        

    
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
    
    