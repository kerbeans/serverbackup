


import  torchvision.utils as  tvu
import torch 
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer,CLIPVisionModel
from data_utils import FashionIQDataset , FashionIQsmall
from torch.utils.data import DataLoader
from FashionImg2ImgPipeline import FashionImg2ImgPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from func_utils import model_log
import json
import os
import torch.nn.functional as F
SD_MODEL='runwayml/stable-diffusion-v1-5' 
MODEL_PATH ='model/baseLine_128_1e-06_40'#'model/info_nce_dc_128_1e-06_40'#
##'model/info_nce_128_1e-06_20'#'model/grid/kldis_16_1e-07_10'
#model/info_nce_128_1e-06_20

def multi_collate_fn(batch):
    sample={}
    sample['ref_name']=[i['ref_name'] for i in batch]
    sample['tar_name']=[i['tar_name'] for i in batch]
    sample['gen_input']=[i['gen_input'] for i in batch]
    return sample
        
        



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
    

from tqdm import tqdm
from PIL import Image
import numpy as np
def comparible_matrix(output_path,device,batchsize=8,using_clip=None):
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
    
    
    
def recall(image_folder,compare_matrix,device,batch_size=32,result=None,multi=0,comments=""):
    elog={}
    elog['compare_matrix']=compare_matrix
    elog['image_folder']=image_folder
    elog['comments']=comments
    if isinstance(type(compare_matrix),type(str)):
        compare_matrix=np.load(compare_matrix,allow_pickle=True).item()
    name_dict={}
    matrix=[]
    for i,(name,emb) in enumerate(compare_matrix.items()):
        name_dict[name]=i
        matrix.append(emb)
    matrix=np.stack(matrix)
    print(f' choice {multi}')
    if False  and os.path.exists(f'{image_folder}.npy'):
        gen_matrix=list(np.load(f'{image_folder}.npy',allow_pickle=True))
        print(type(gen_matrix))

    elif multi==0 or multi ==2:
        fashionIQeval= FashionIQsmall(image_folder,dim=224,multi=multi)
        small_dataloader=DataLoader(fashionIQeval,batch_size=batch_size,drop_last=False,num_workers=16)
        gen_matrix=[]
        with torch.no_grad():
            lower_encoder = torch.load('l_img_encoder.ckpt').to(device)
            upper_encoder = torch.load('u_img_encoder.ckpt').to(device)
            for sample in tqdm(small_dataloader):
                mediate ,_=lower_encoder(sample['gen_input'].to(device))
                features = upper_encoder(mediate).detach().to('cpu').numpy()
                for i,name in enumerate(sample['ref_name']):
                    if not sample['tar_name'][i] in name_dict:
                        continue 
                    temp={}
                    temp['ref']=name
                    temp['tar']=sample['tar_name'][i]
                    temp['feature']=features[i]
                    gen_matrix.append(temp)
        elog['matix length']=len(gen_matrix)
    elif multi ==1:
        fashionIQeval= FashionIQsmall(image_folder,dim=224,multi=multi)
        small_dataloader=DataLoader(fashionIQeval,batch_size=batch_size,drop_last=False,num_workers=4,collate_fn=multi_collate_fn)
        gen_matrix=[]
        print('using average')
        with torch.no_grad():
            lower_encoder = torch.load('l_img_encoder.ckpt').to(device)
            upper_encoder = torch.load('u_img_encoder.ckpt').to(device)
            for sample in tqdm(small_dataloader):
                # gen_input=[i for i in sample['gen_input'].chunk(batch_size)]
                mediates=[]
                # if len(sample['gen_input'])== 1:
                #     print('error length ',len(sample['gen_input']))
                for i in sample['gen_input']:
                    mediate ,_=lower_encoder(i.to(device))# 20*2000*14*14
                    features = upper_encoder(F.normalize(mediate,p=2,dim=-1))#----
                    #print(features.shape)
                    mediates.append(features.mean(0))#----
                mediate=torch.stack(mediates,dim=0)
                features=torch.stack(mediates,dim=0).detach().to('cpu').numpy()
                
                
                #mediate ,_=lower_encoder(sample['gen_input'].mean(1).to(device))
                #features = upper_encoder(mediate).detach().to('cpu').numpy()
                
                
                for i,name in enumerate(sample['ref_name']):
                    if not sample['tar_name'][i] in name_dict:
                        continue 
                    temp={}
                    temp['ref']=name
                    temp['tar']=sample['tar_name'][i]
                    temp['feature']=features[i]
                    gen_matrix.append(temp)
    
    elif multi == -1:
        fashionIQeval= FashionIQsmall(image_folder,dim=224,multi=multi)
        small_dataloader=DataLoader(fashionIQeval,batch_size=batch_size,drop_last=False,num_workers=16)
        gen_matrix=[]
        with torch.no_grad():
            vision_model=torch.load(f'{MODEL_PATH}/clip-vit-base-patch32.pth').to(device)
            #vision_model = CLIPVisionModel.from_pretrained(MODEL_PATH,subfolder='clip-vit-base-patch32').to(device)
            for sample in tqdm(small_dataloader):
                features = vision_model(pixel_values=sample['gen_input'].to(device))[1].to('cpu').numpy()
                for i,name in enumerate(sample['ref_name']):
                    if not (sample['tar_name'][i] in name_dict):
                        continue 
                    temp={}
                    temp['ref']=name
                    temp['tar']=sample['tar_name'][i]
                    temp['feature']=features[i]
                    gen_matrix.append(temp)
        elog['matix length']=len(gen_matrix)
    #np.save(f'{image_folder}.npy',gen_matrix)
        #gen =[ref,tar,feature]
    #print(f'gt table {matrix.shape}, gen table {gen_matrix}')
    name_ref_dict={}
    name_tar_dict={}
    gen_matrix_=[]
    for i ,triplets in enumerate(gen_matrix):
        name_ref_dict[i]=triplets['ref']
        name_tar_dict[i]=triplets['tar']
        gen_matrix_.append(triplets['feature'])
    gen_matrix=np.stack(gen_matrix_)
    dot_matrix=np.dot(gen_matrix,matrix.T)
    dot_score=dot_matrix.copy()
    dot_matrix=np.argsort(dot_matrix,axis=1)[:,::-1]
    recall_table=np.zeros(dot_matrix.shape)
    for i in range(dot_matrix.shape[0]):
        recall_table[i,name_dict[name_tar_dict[i]]]=1
    recall_table=np.array(list(map(lambda x,y:y[x],dot_matrix,recall_table)))
    size=recall_table.shape[0]
    if result:
        np.save(f'{result}.npy',[recall_table,dot_matrix,dot_score,name_ref_dict,name_tar_dict])
    record=[{f'recall@{i}':np.sum(recall_table[:,:i])/size*100} for i in [1,5,10,50]]
    elog['matrix']=record
    elog['size']=size
    model_log('recall_matrix.log',elog)
    
    
    print(record,size)
    
    
        

def dummy(images, **kwargs):
    return images,False





import cv2




def SiftMatching(generated_file='validation/full_net_NCE',lib_file='data/images',rate=0.95):
    val_triplets=[]
    elog={}
    gt_colum_feature=[]#{imgname} {feature}
    gt_colum_index={}#{imgname}{index}
    pointing_to_target=[]
    gen_imgs=[]
    for root,dir,files in os.walk(generated_file):
        for file in files:
            if 'jpg' in file:
                gen_imgs.append(file)
    gen_imgs=gen_imgs[:200]
    sift=cv2.SIFT_create()
    matcher=cv2.BFMatcher()
    gen_features=[]
    target_id=0
    for e,i in enumerate(gen_imgs):
        gimg=cv2.imread(f"{generated_file}/{i}",1)
        target_img=i.replace('.jpg',"").split("_")[-1]
        kp1,des1=sift.detectAndCompute(gimg,None)
        if target_img not in gt_colum_index.keys():
            gt_colum_index[target_img]=target_id
            tar_img=cv2.imread(f'data/images/{target_img}.jpg',1)
            kp2,des2=sift.detectAndCompute(tar_img,None)
            gt_colum_feature.append([kp2,des2])
            target_id+=1
            
        gen_features.append([kp1,des1])
        pointing_to_target.append(gt_colum_index[target_img])
    cool_matrix=np.zeros((len(gen_features),len(gt_colum_feature)),dtype=float)#
    for i in range(cool_matrix.shape[0]):
        for j in range(cool_matrix.shape[1]):
            good_match=0
            try:
                raw_match = matcher.knnMatch(gen_features[i][1], gt_colum_feature[j][1], k = 2)
                for m1,m2 in raw_match:
                    if m1.distance<m2.distance*rate:
                        good_match+=1
            except:
                good_match=0
            cool_matrix[i,j]=good_match
    
    cool_matrix=np.argsort(cool_matrix,axis=1)[:,::1]
    recall_table=np.zeros(cool_matrix.shape)
    for i in range(cool_matrix.shape[0]):
        recall_table[i,np.where(cool_matrix[i,:]==pointing_to_target[i])]=1
    recall_table=np.array(list(map(lambda x,y:y[x],cool_matrix,recall_table)))
    record=[{f'recall@{i}':np.sum(recall_table[:,:i])/cool_matrix.shape[0]*100} for i in [1,5,10,50]]
    elog['image_folder']=generated_file

    elog['size']=cool_matrix.shape[0]
    elog['rate']=rate
    elog['matrix']=record
    model_log('sift.log',elog)
    print(record)
    # sample['ref_name']=self.imgs[index].split("_")[0]
    # sample['tar_name']=self.imgs[index].replace(".jpg","").split("_")[-1]
    # sample['gen_input']=self.img_transformer_for_evaluation(PIL.Image.open(f"{self.file_path}/{self.imgs[index]}"))
    def transfrom_clip(path):
        pass
    


def pseudo_matching(model_path,compare_matrix='clip_model/compare_matrix_clip.npy',batch_size=16,device='cuda'):
    
    clip_text_model=CLIPTextModel.from_pretrained(model_path,subfolder='text_encoder').to(device)
    #clip_img_model=CLIPVisionModel.from_pretrained(model_path,subfolder='clip-vit-base-patch32').to(device)
    #clip_text_model=CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    clip_img_model=CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    fashioniq= FashionIQDataset('val',['dress','shirt','toptee'],
                                tokenizer=CLIPTokenizer.from_pretrained(model_path,subfolder="tokenizer"),
                                preprossor=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32'),
                                display=False,
                                dim=256
                                )
    if isinstance(type(compare_matrix),type(str)):
        compare_matrix=np.load(compare_matrix,allow_pickle=True).item()
    name_dict={}
    matrix=[]
    for i,(name,emb) in enumerate(compare_matrix.items()):
        name_dict[name]=i
        matrix.append(emb)
    matrix=np.stack(matrix)
    gen_matrix=[]
    fashionloader=DataLoader(fashioniq,batch_size=batch_size,drop_last=True,num_workers=4)
    with torch.no_grad():
        for sample in tqdm(fashionloader):
            # print(type(sample['ref_i']))
            # print(sample['ref_i'].shape)
            img_features=clip_img_model(pixel_values=sample['ref_i'].to(device))[1]
            text_features=clip_text_model(sample['caption'].to(device))[1]
            #print(f'shape img {img_features.shape}, text {text_features.shape}')
            compose_feature=img_features+text_features
            for i ,name in enumerate(sample['ref_name']):
                if not (sample['target_name'][i] in name_dict) :
                    continue
                temp={}
                temp['ref']=name
                temp['tar']=sample['target_name'][i]
                temp['feature']=compose_feature[i].detach().to('cpu').numpy()
                gen_matrix.append(temp)
    name_ref_dict={}
    name_tar_dict={}
    gen_matrix_=[]
    for i ,triplets in enumerate(gen_matrix):
        name_ref_dict[i]=triplets['ref']
        name_tar_dict[i]=triplets['tar']
        gen_matrix_.append(triplets['feature'])
    gen_matrix=np.stack(gen_matrix_)
    dot_matrix=np.dot(gen_matrix,matrix.T)
    dot_score=dot_matrix.copy()
    dot_matrix=np.argsort(dot_matrix,axis=1)[:,::-1]
    recall_table=np.zeros(dot_matrix.shape)
    for i in range(dot_matrix.shape[0]):
        recall_table[i,name_dict[name_tar_dict[i]]]=1
    recall_table=np.array(list(map(lambda x,y:y[x],dot_matrix,recall_table)))
    print(recall_table.shape)
    size=recall_table.shape[0]
    elog={}
    record=[{f'recall@{i}':np.sum(recall_table[:,:i])/size*100} for i in [1,5,10,50]]
    elog['matrix']=record
    elog['size']=size
    model_log('recall_matrix.log',elog)
    print(record)
if __name__ =='__main__':
    #SiftMatching()    
    #comparible_matrix("clip_model/compare_matrix_clip_unfinetuned.npy",'cuda:3',4,using_clip=MODEL_PATH)
    # recall('validation/regenerate','compare_matrix.npy','cuda:3',result='log/20_combined',
    #        batch_size=4,multi=0,comments="clip_unfintuned")
    #pseudo_matching(MODEL_PATH,compare_matrix='clip_model/compare_matrix_clip_unfinetuned.npy',device='cuda:3')
    # recall('validation/dual_clip','compare_matrix.npy','cuda:3',result='log/result_tb_info',batch_size=32,multi=0)
    
    #MODEL_PATH="model/info_unet++_64_1e-06_20"
    #---------------------------
    validation_image("baseline",MODEL_PATH,batch=1,device='cuda',flag='img2img',dim=256,overwrite=False)(flag=None,image_num=20)
    #-------------------------------
    # pipe=FashionImg2ImgPipeline.from_pretrained(MODEL_PATH).to('cuda:3')
    # pipe.safety_checker=dummy
    # img=Image.open('self.jpg')
    # img=pipe.mycall(
    #     "more longer sleeves",
    #     img
    # ).images[0]

    # img.save('self_gen.jpg')