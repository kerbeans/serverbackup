import os 
from transformers import CLIPFeatureExtractor ,CLIPTokenizer, CLIPVisionModel
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from dataset.FashionIQDataset import FashionIQDataset
from diffusers import StableDiffusionImg2ImgPipeline
from module.pipeline import FashionImg2ImgPipeline



class validation_image():
    def __init__(self,val_path,model_path,device,flag="PIX2PIX",dim=256,batch=1,overwrite=False):
        self.val_path=f"validation/{val_path}"
        self.model_path=model_path
        self.device=device
        self.flag=flag
        self.batch_size=batch
        self.dim=dim
        self.overwrite=overwrite
        if not os.path.exists("validation"):
            os.mkdir("validation")
        if not os.path.exists(f"validation/{val_path}"):
            os.mkdir(f"validation/{val_path}")
    def dummy(self,images,**kwargs):
        return  images, False
    def prepare_model(self):
        if self.flag=="PIX2PIX":# run on ldm
            self.model=FashionImg2ImgPipeline.from_pretrained(self.model_path).to(self.device)
        elif self.flag=="dc":# run on ldmv100
            self.model=FashionImg2ImgPipeline.from_pretrained(self.model_path).to(self.device)
        elif self.flag=='img2img':
            self.model=StableDiffusionImg2ImgPipeline.from_pretrained(self.model_path).to(self.device)
        self.model.safety_checker =self.dummy
        
    def pipeline_call(self,sample,guidance_scale=7.5,num_steps=50):
        if self.flag=="PIX2PIX":
            return self.model.pix2pix(
                sample['cap_word'],
                init_image=sample['ref'].to(self.device)
            ).images
        elif self.flag=='dc':
            return self.model.mycall(
                sample['cap_word'],#
                #[""]*len(sample['cap_word']),
                init_image=sample['ref'].to(self.device),
                out_dim=self.dim,
                init_image_i=sample['ref_i'].to(self.device),
                init_overwrite=self.overwrite,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps
            ).images
        elif self.flag =='img2img':
            return self.model(
                sample['cap_word'][0],
                Image.open(f'data/images/{sample["ref_name"][0]}.jpg').convert('RGB')
            ).images

    @torch.no_grad()
    def __call__(self,flag=None,image_num=1):
        fashioniq= FashionIQDataset('val',['dress','shirt','toptee'],
                                    tokenizer=CLIPTokenizer.from_pretrained(self.model_path,subfolder="tokenizer"),
                                    preprossor=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32'),
                                    display=True,
                                    dim=self.dim
                                    )
        #fashioniq,_ =torch.utils.data.random_split(fashioniq,[1000,len(fashioniq)-1000])
        self.prepare_model()
        total_step=len(fashioniq)
        FashionLoader=DataLoader(fashioniq,batch_size=self.batch_size,drop_last=True)
        if flag is None:   
            for step, sample in tqdm(enumerate(FashionLoader)):
                imgs=self.pipeline_call(sample)
                for i, img in enumerate(imgs):
                    img.save(f"{self.val_path}/{sample['ref_name'][i]}_{sample['target_name'][i]}.jpg")
        elif flag == 'multi':
            for step, sample in tqdm(enumerate(FashionLoader)):
                for roun in range(image_num):
                    if os.path.exists(f"{self.val_path}/{sample['ref_name'][0]}_{roun}_{sample['target_name'][0]}.jpg"):
                        continue
                    imgs=self.pipeline_call(sample)  
                    for i, img in enumerate(imgs):
                        # if not os.path.exists(f"{self.val_path}/{sample['ref_name'][i]}_{sample['target_name'][i]}"):
                        #     os.mkdir(f"{self.val_path}/{sample['ref_name'][i]}_{sample['target_name'][i]}")
                        img.save(f"{self.val_path}/{sample['ref_name'][i]}_{roun}_{sample['target_name'][i]}.jpg")
        elif flag== 'scale':
            scales=[4.5,7.5,10.5,15.5]
            for step, sample in tqdm(enumerate(FashionLoader)):
                for roun,scale in enumerate(scales):
                    if os.path.exists(f"{self.val_path}/{sample['ref_name'][0]}_{roun}_{sample['target_name'][0]}.jpg"):
                        continue
                    imgs=self.pipeline_call(sample,scale)  
                    for i, img in enumerate(imgs):
                        # if not os.path.exists(f"{self.val_path}/{sample['ref_name'][i]}_{sample['target_name'][i]}"):
                        #     os.mkdir(f"{self.val_path}/{sample['ref_name'][i]}_{sample['target_name'][i]}")
                        img.save(f"{self.val_path}/{sample['ref_name'][i]}_{roun}_{sample['target_name'][i]}.jpg")  
        elif flag == 'inference step':
            inf_steps=[20,30,40,50,80]
            for step, sample in tqdm(enumerate(FashionLoader)):
                for roun,inf_step in enumerate(inf_steps):
                    if os.path.exists(f"{self.val_path}/{sample['ref_name'][0]}_{roun}_{sample['target_name'][0]}.jpg"):
                        continue
                    imgs=self.pipeline_call(sample,num_steps=inf_step)  
                    for i, img in enumerate(imgs):
                        # if not os.path.exists(f"{self.val_path}/{sample['ref_name'][i]}_{sample['target_name'][i]}"):
                        #     os.mkdir(f"{self.val_path}/{sample['ref_name'][i]}_{sample['target_name'][i]}")
                        img.save(f"{self.val_path}/{sample['ref_name'][i]}_{roun}_{sample['target_name'][i]}.jpg")  
        elif flag =='prompt':
            prompt=['a photo of dress that',
                    'a photo of fashion dress that']
            for step, sample in tqdm(enumerate(FashionLoader)):
                for roun,preflex in enumerate(prompt):
                    if os.path.exists(f"{self.val_path}/{sample['ref_name'][0]}_{roun}_{sample['target_name'][0]}.jpg"):
                        continue
                    sample['cap_word']=[f"{preflex} {i}" for i in sample['cap_word']]
                    imgs=self.pipeline_call(sample)  
                    for i, img in enumerate(imgs):
                        # if not os.path.exists(f"{self.val_path}/{sample['ref_name'][i]}_{sample['target_name'][i]}"):
                        #     os.mkdir(f"{self.val_path}/{sample['ref_name'][i]}_{sample['target_name'][i]}")
                        img.save(f"{self.val_path}/{sample['ref_name'][i]}_{roun}_{sample['target_name'][i]}.jpg")  
                        
def comparible_matrix(output_path,MODEL_PATH,device,batchsize=8,using_clip=None):
    gt_matrix={}
    if using_clip is not None:
        fashioniq= FashionIQDataset('val',['dress','shirt','toptee'],light=True,
                                    dim=224,preprossor=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32'))
        fashioniq_loader=DataLoader(fashioniq,batch_size=batchsize,drop_last=False,num_workers=16)
        with torch.no_grad():
            vision_model=CLIPVisionModel.from_pretrained(MODEL_PATH,subfolder='clip-vit-base-patch32').to(device)#
            #torch.load(f'{MODEL_PATH}/clip-vit-base-patch32.pth').to(device)#
            for sample in tqdm(fashioniq_loader):
                features = vision_model(pixel_values=sample['img_input'].to(device))[1].to('cpu').numpy()
                for i, name in enumerate(sample['img_name']):
                    gt_matrix[name]=features[i]
            
    else:
        fashioniq= FashionIQDataset('val',['dress','shirt','toptee'],light=True,dim=224)
        fashioniq_loader=DataLoader(fashioniq,batch_size=batchsize,drop_last=False,num_workers=16)
        with torch.no_grad():
            lower_encoder = torch.load('l_img_encoder.ckpt').to(device)
            upper_encoder = torch.load('u_img_encoder.ckpt').to(device)
            for sample in tqdm(fashioniq_loader):
                mediate ,_=lower_encoder(sample['img_input'].to(device))
                features = upper_encoder(mediate).detach().to('cpu').numpy()
                for i,name in enumerate(sample['img_name']):
                    gt_matrix[name]=features[i]
                #gt_matrix[sample['tar_name']]=features
    np.save(output_path,gt_matrix)