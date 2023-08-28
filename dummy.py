from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
# from transformers import CLIPTextModel, CLIPVisionModel
from dataset.FashionIQDataset import FashionIQDataset, FashionIQDataset_light,collate_fn_light
import torch
import sys
model_path='model/info_nce_dc_128_1e-06_40'

from torchvision import make_dot

def caption_len():
    fiq=FashionIQDataset(split='val',dress_types=['shirt','dress','toptee'])
    print(max([len(i['captions'][0].split(' '))+len(i['captions'][1].split(' ')) for i in fiq.triplets]))


def seeseeDataloader():
    fiq=FashionIQDataset_light(split='train',dress_types=['shirt','dress','toptee'])
    fiqdl=torch.utils.data.DataLoader(fiq,batch_size=4,drop_last=True,collate_fn=collate_fn_light)
    for step,batch in enumerate(fiqdl):
        if step==0:
            print(batch)
            return 




def seeseemodel():
    
    fiq=FashionIQDataset(split='val',dress_types=['shirt','dress','toptee'])
    target =set([i['target'] for i in fiq.triplets])
    ref=set([i['candidate'] for i in fiq.triplets])
    hinge= list(target&ref)
    
    
    def sequentialQuery(currentquery,database): # database = triplets 
        step=0
        while(len(currentquery)>0):
            res=[]
            currentquery=list(set(currentquery))
            for i in currentquery:
                for j in database:
                    if i==j['candidate']:
                        res.append(j['target'])
            currentquery=res
            print(f'cycle step {step}, query length {len(currentquery)}')
            step+=1
            
    
    def display(index,data_s):
        img=hinge[index]
        astarget=[]
        asreference=[]
        for i in data_s.triplets:
            if i['candidate']==img:
                asreference.append(i)
            elif i['target']==img:
                astarget.append(i)
        return astarget,asreference
    # print(display(2,fiq))
    sequentialQuery([i['target'] for i in fiq.triplets],fiq.triplets)
    
    
    
    return 
    time = torch.randint(0, 2, (1,)).long()
    encod=torch.randn((1,77,768))
    x=torch.randn((1,4,32,32))
    out=unet(x,time,encoder_hidden_states=encod).sample
    #out=vae.encode(torch.randn((1,3,256,256))).latent_dist.sample()
    g=make_dot(out)
    g.render(filename='vae',view=False,format='pdf')
    # print(unet)
    # vae=AutoencoderKL.from_pretrained(model_path,subfolder='vae')
    # print(vae)
import numpy as np
import os
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision
import PIL
if __name__ =='__main__':
    # pipe=StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
    # cvm=CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
    # pipe.save_pretrained('model/sd1v5')
    # cvm.save_pretrained('model/sd1v5/clip-vit-base-patch32')
    
    img=[]
    for root,folder,files in os.walk('data/pose'):
        img=[i for i in files if '.jpg' in i]
    out_dir='pose_output'
    if os.path.exists(out_dir):
        os.mkdir(out_dir)
    img=img[np.random.randint(0,70000,10)]
    for im in img:
        img1=read_image(f'data/images/{im}')
        img2=read_image(f'data/pose/{im}')
        grid=make_grid([img1,img2],nrow=2)
        grid=torchvision.transforms.ToPILImage()(grid)
        grid.save(f'{out_dir}/{im}')