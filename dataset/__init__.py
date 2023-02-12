import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel , CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
import json
# from .FashionIQDataset import FashionIQDataset,FashionIQDataset_dc,FashionIQDataset_eval

def multi_collate_fn(batch):
    sample={}
    sample['ref_name']=[i['ref_name'] for i in batch]
    sample['tar_name']=[i['tar_name'] for i in batch]
    sample['gen_input']=[i['gen_input'] for i in batch]
    return sample


def preprocess(image,dim=None):
    if dim :
        image = image.resize((dim,dim), resample=Image.LANCZOS)
    else :
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = np.array(image).astype(np.float32) / 255.0
    if len(image.shape)<3:
        image=np.stack((image,)*3,-1)
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0



def choose_best(file_list,folder,diffusion_path=None,device='cuda'):
    if diffusion_path is None:
        diffusion_path='runwayml/stable-diffusion-v1-5'
        vision_model=CLIPVisionModel.from_pretrained(f'openai/clip-vit-base-patch32').to(device)
        text_encoder=CLIPTextModel.from_pretrained(f'{diffusion_path}', subfolder="text_encoder").to(device)
        text_tokenizer=CLIPTokenizer.from_pretrained(f'{diffusion_path}', subfolder="tokenizer")
    else:
        vision_model=CLIPVisionModel.from_pretrained(f'{diffusion_path}/clip-vit-base-patch32').to(device)
        text_encoder=CLIPTextModel.from_pretrained(f'{diffusion_path}', subfolder="text_encoder").to(device)
        text_tokenizer=CLIPTokenizer.from_pretrained(f'{diffusion_path}', subfolder="tokenizer")
    img_preprocessor=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
    triplets=[]
    res=[]
    for dress_type in ['dress','shirt','toptee']:
        with open('data' / 'captions' / f'cap.{dress_type}.val1.json') as f:
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
        pixel_value_gen=torch.stack([torch.from_numpy(img_preprocessor(Image.open(f'{folder}/{j}.jpg'))['pixel_values'][0]).to(device) for j in i],dim=0)
        i_m_gen=vision_model(pixel_values=pixel_value_gen)[1]# b* f
        pixel_value_ref=torch.from_numpy(img_preprocessor(Image.open(f'data/images/{ref_img_name}.jpg'))['pixel_values'][0]).to(device).unsqueeze(0)
        i_m_ref=vision_model(pixel_values=pixel_value_ref)[1]
        compose=i_m_ref+t_m# 1*f
        table= i_m_gen@compose.T
        table=table.detach().to('cpu').numpy()
        #print(table.shape,table.max(0),np.argmax(table,0))#
        res.append(i[np.argmax(table,0)[0]]) 
        #print('res is ',res)
    print('choosen finished')   
    return res