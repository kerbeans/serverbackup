from typing import List , Union ,Dict
from torchvision import transforms
from transformers import CLIPTokenizer,CLIPFeatureExtractor
from PIL import Image
import os 
import torch

from .base_dataset import BaseDataset
from dataset import preprocess,choose_best

def collate_fn_light(data):
    batch:Dict[List]={}
    batch['cap_word']=[]
    batch['ref']=[]
    batch['target']=[]
    batch['caption']=[]
    for unit in data:
        # print(f'unit  {unit}, {type(unit)}')
        batch['cap_word'].append(unit['cap_word'])
        batch['ref'].append(unit['ref'])
        batch['target'].append(unit['target'])
        batch['caption'].append(unit['caption'])
    return batch 

class FashionIQDataset(BaseDataset):
    '''
        for original diffusion, return ref,caption,tar
    '''
    def __init__(self, tokenizer:CLIPTokenizer=None,dim=None,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer=tokenizer
        self.dim=dim
    def __getitem__(self, index):
        try:
            image_captions = self.triplets[index]['captions']
            example={}
            example['cap_word']=str(image_captions[0]+', '+image_captions[1])
            if self.tokenizer is not None:
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
                reference_image_path = f"{self.data_path}/images/{reference_name}.jpg"
                target_name = self.triplets[index]['target']
                target_image_path = f"{self.data_path}/images/{target_name}.jpg" 
                reference_image = preprocess(Image.open(reference_image_path).convert('RGB'),self.dim)
                target_image =preprocess(Image.open(target_image_path).convert("RGB"),self.dim)
                example['ref']=reference_image.squeeze(0)
                example['target']=target_image.squeeze(0)
            elif self.split == 'test':
                reference_image_path = f"{self.data_path}/images/{reference_name}.jpg"
                reference_image = preprocess(Image.open(reference_image_path).convert("RGB"),self.dim)
                example['ref']=reference_image.squeeze(0)

            return example
        except Exception as e:
            print(e)
    
    
    
    
class FashionIQDataset_dc(FashionIQDataset):
    '''
        for clip img encoder, return ref,ref_i,caption, tar,tar_i
    '''
    def __init__(self, split: str, dress_types: List[str], device,tokenizer: CLIPTokenizer = None, dim=None, 
                 CLIP_preprocess:CLIPFeatureExtractor=None) -> None:
        super().__init__(split, dress_types, tokenizer, dim)
        self.CLIP_preprocess=CLIP_preprocess
        if self.CLIP_preprocess is None:
            self.CLIP_preprocess = CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
    def __getitem__(self, index):
        try:
            image_captions = self.triplets[index]['captions']
            example={}
            example['cap_word']=str(image_captions[0]+', '+image_captions[1])
            if self.tokenizer is not None:
                image_captions=self.tokenizer(
                    example['cap_word'],
                    padding='max_length',
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
                example['caption']=image_captions
            reference_name = self.triplets[index]['candidate']
            example['ref_name']=reference_name
            if self.split == 'train'or self.split=='val':
                reference_image_path = f"{self.data_path}/{reference_name}.jpg"
                target_name = self.triplets[index]['target']
                target_image_path = f"{self.data_path}/{target_name}.jpg" 
                reference_image_dim =self.CLIP_preprocess(Image.open(reference_image_path).convert("RGB"))['pixel_values'][0]
                reference_image = preprocess(Image.open(reference_image_path).convert('RGB'),self.dim)
                target_image =preprocess(Image.open(target_image_path).convert("RGB"),self.dim)
                example['ref']=reference_image.squeeze(0)
                example['ref_i']=reference_image_dim
                example['target']=target_image.squeeze(0)
                example['tar_i']=self.CLIP_preprocess(Image.open(target_image_path))['pixel_values'][0]
            elif self.split == 'test':
                reference_image_path = f"{self.data_path}/{reference_name}.jpg"
                reference_image = preprocess(Image.open(reference_image_path).convert("RGB"),self.dim)
                reference_image_dim =self.CLIP_preprocess(Image.open(reference_image_path).convert("RGB"))['pixel_values'][0]
                example['ref_i']=reference_image_dim.squeeze(0)
                example['ref']=reference_image.squeeze(0)

            return example
        except Exception as e:
            print(f"{e} error catched in dataloader")
        
    
    
class FashionIQDataset_eval(BaseDataset):
    '''
        given img folder path ref_{index/}_tar.jpg or data/images/
        return img,img_name for retreval head 
        
        return gen,tar || gen, ref 
    '''
    def __init__(self, split: str=None, dress_types: List[str]=None,folder_path=None,retrival_preprocess=None,
                 strategy:Union[str,int]=None,device='cuda',
                 model_path=None
                 ) -> None:
        super().__init__(split, dress_types)
        self.strategy=strategy
        if folder_path is None:
            candidate= set([i['candidate'] for i in self.triplets ])
            target =set([i['target'] for i in self.triplets ])
            self.triplets =list(candidate|target) 
        else:
            self.data_path=folder_path
            self.triplets=[]
            for root,dir,files in os.walk(self.data_path):
                for file in files:
                    if 'jpg' in file:
                        self.triplets.append(file.replace('.jpg',""))
            try:
                if isinstance(strategy,str) and strategy=='best':
                    self.triplets=choose_best(self.triplets,self.data_path,diffusion_path=model_path,device=device)
                elif isinstance(strategy,str) and strategy == 'average':
                    type_list=set([f"{i.split('_')[0]}_{i.split('_')[-1]}" for i in self.triplets])  
                    imgs_type=[]
                    for i in type_list:  
                        imgs_type.append([k for k in self.triplets if 
                          (k.split('_')[0]==i.split('_')[0])and(k.split('_')[-1]==i.split("_")[-1])])
                    self.triplets=imgs_type
                elif isinstance(strategy,int):
                    self.triplets=[ i for i in self.triplets if i.split("_")[1]==strategy]
                if len(self.triplets)==0:
                    raise Exception
            except Exception as e :
                print(f'strategy error {e}, strategy {strategy}, folder {self.data_path}')
                
        self.retrival_preprocess=retrival_preprocess
        if self.retrival_preprocess is None:
            IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
            self.retrival_preprocess=transforms.Compose([transforms.Resize((self.dim, self.dim)), transforms.ToTensor(),
                               transforms.Normalize(**IMAGENET_STATS)])
    def __getitem__(self, index):
            if (self.strategy is None ) or(isinstance(self.strategy,str) and (self.strategy=='best') ) :
                example={}
                example['ref_name']=self.triplets[index].split("_")[0]
                example['tar_name']=self.triplets[index].replace(".jpg","").split("_")[-1]
                example['gen_input']=self.retrival_preprocess(Image.open(f"{self.data_path}/{self.triplets[index]}.jpg").convert("RGB"))
            elif isinstance(self.strategy,str) and (self.strategy =='average'):
                example={}
                example['ref_name']=self.triplets[index][0].split("_")[0]
                example['tar_name']=self.triplets[index][0].replace(".jpg","").split("_")[-1]
                example['gen_input']=[]
                for i in self.triplets[index]:
                    example['gen_input'].append(self.retrival_preprocess(Image.open(f"{self.file_path}/{i}.jpg")).convert("RGB"))
                example['gen_input']=torch.stack(example['gen_input'],dim=0)
            return example
        
        
        
class FashionIQDataset_light(BaseDataset):
    '''
        for original diffusion, return ref,caption,tar, type(Image.PIL)
    '''
    def __init__(self, split: str, dress_types: List[str],tokenizer:CLIPTokenizer=None,dim=None) -> None:
        super().__init__(split, dress_types)
        self.tokenizer=tokenizer
        self.dim=dim
    def __getitem__(self, index):
        try:
            image_captions = self.triplets[index]['captions']
            example={}
            example['cap_word']=str(image_captions[0]+', '+image_captions[1])
            if self.tokenizer is not None:
                image_captions=self.tokenizer(
                    example['cap_word'],
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
                example['caption']=image_captions
            reference_name = self.triplets[index]['candidate']
            if self.split == 'train'or self.split=='val':
                reference_image_path = f"{self.data_path}/{reference_name}.jpg"
                target_name = self.triplets[index]['target']
                target_image_path = f"{self.data_path}/{target_name}.jpg" 
                reference_image = Image.open(reference_image_path).convert('RGB')
                target_image =Image.open(target_image_path).convert("RGB")
                example['ref']=reference_image
                example['target']=target_image
            elif self.split == 'test':
                reference_image_path = f"{self.data_path}/{reference_name}.jpg"
                reference_image = Image.open(reference_image_path).convert("RGB")
                example['ref']=reference_image.squeeze(0)
            return example
        except Exception as e:
            print(e)
            


class FashionIQDataset_combinedEmb_poseControl(BaseDataset):
    def __init__(*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    @staticmethod
    
    
    
    def __getitem__(self, idx):
        '''
        returns 
            captions [B,3,scentence] # after using embedding  [B,3 70,768]
            ref_image [B,3,256,256] # after VAEKL [B,4,32,32]
            target_image [B,3,256,256] after VAEKL [B,4,32,32]
            pose_image[B,3,256,256] # to the control_net part
            infos [{tar_dir:str,ref_dir:str,captions:[str]}]*B
        '''
        item = self.triplets[idx]   
        example={}
        example["infos" ]= item
        source_path = f"{self.data_path}/images/{item['candidate']}"     # source
        target_path = f"{self.data_path}/images/{item['target']}"   # target
        pose_path =f"{self.data_path}/pose/{item['target']}"
        
        captions= item['captions']
        return example


if '__main__' == __name__:
    dt=FashionIQDataset_combinedEmb_poseControl('train',['dress'])
    from torch.utils.data import DataLoader
    dl=DataLoader(dt)
    for i in dl:
        print(i)
        exit()
        
        
        