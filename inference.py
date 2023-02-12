import torch
import PIL
import numpy as np
from pathlib import Path

from diffusers import StableDiffusionImg2ImgPipeline,AutoencoderKL,DDPMScheduler,StableDiffusionPipeline
from data_utils import FashionIQDataset ,collate_fn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer,CLIPFeatureExtractor, CLIPTextModel
from FashionImg2ImgPipeline import FashionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import torchvision.utils as tvu
from func_utils import model_log
from pytorch_fid.fid_score import calculate_fid_given_paths
from diffusers.schedulers import DDIMScheduler
#from lib_vd import UNet2DConditionModel
from diffusers import UNet2DConditionModel
base_path= Path(__file__).absolute().parents[1].absolute()
import os 
#from lib_vd import UNet2DConditionModel
import json



model_path = 'model/attn_vd_ne_32_1e-05_5'#"CompVis/stable-diffusion-v-1-4-original"#"model/text-inversion-model"
ouput_dir="output/vd32_ep5"# attention2:15000*16+6000*32 =27
#model_path="model/unfreeze_s30000"
img_size=(256,256)
def main():     
        def dummy(images, **kwargs):
                return images, False

        # # ---------------M1--------------
        # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, 
        #  use_auth_token='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',
        # revision="fp16", torch_dtype=torch.float16
        # ).to("cuda:6")
        
        # ---------------M2--------------
        pipe =FashionImg2ImgPipeline.from_pretrained(model_path,
                use_auth_token='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',
                revision='fp16',torch_dtype=torch.float32
        ).to('cuda:0')
        
        # ---------------M3--------------
        # pipe = FashionImg2ImgPipeline(
        #     text_encoder=CLIPTextModel.from_pretrained(model_path,subfolder='text_encoder'),
        #     vae=AutoencoderKL.from_pretrained(model_path,subfolder='vae'),
        #     unet=UNet2DConditionModel.from_pretrained(model_path,subfolder='unet'),
        #     tokenizer=CLIPTokenizer.from_pretrained(model_path,subfolder='tokenizer'),
        #     scheduler=DDPMScheduler.from_config(model_path, subfolder="scheduler"),
        #     safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        # ).to('cuda')
        #test-----------------------
        pipe.safety_checker = dummy
        
        # img=pipe("",guidance_scale=0,num_inference_steps=50,init_image=torch.randn((1,3,512,512)),num_images_per_prompt=3).images
        # pred_img=[torch.from_numpy(np.array(i).astype(np.float32)/255.0).permute(2,0,1) for i in img]
        # print(pred_img[0].shape)
        # grid=tvu.make_grid(torch.stack(pred_img,dim=0),nrow=3,padding=0)
        # tvu.save_image(grid,'null_img2img.jpg')
        # return  
        
        
        
        
        fashioniq= FashionIQDataset('val',['dress','shirt','toptee'],tokenizer=CLIPTokenizer,display=True)
        torch.manual_seed(0)
        _,fashioniq=torch.utils.data.random_split(fashioniq,[len(fashioniq)-10,10])
        xx= DataLoader(dataset=fashioniq,batch_size=1,num_workers=4,#multiprocessing.cpu_count(), 
                shuffle=False,collate_fn=collate_fn)
        iterable=False
        
        for i,sample in enumerate(xx):
                print(f"cap {sample['cap_word'][0]}")
                if iterable ==True:
                        pred_img=sample['ref']
                        piclist=[]
                        for caps in sample['cap_word'][0].split(','):
                               # print(f'pred_img shape {pred_img.shape}')
                                img=pipe(f"{caps}",init_image=pred_img,
                                        guidance_scale=7.5, num_inference_steps=50).images[0]
                                pred_img=torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
                                piclist.append(pred_img.squeeze(0)) 
                else :
                        img=pipe(f"{sample['cap_word'][0]}",init_image=sample['ref_i'],
                                guidance_si=3.5, guidance_st=5.5,
                                num_inference_steps=50).images[0]
                        #print(f' sample size {sample["ref"].shape}')
                        # img = pipe.mycall(f"{sample['cap_word'][0]}",init_image=sample['ref_i'],
                        #                   guidance_img=sample['ref'],guidance_si=1.5,guidance_st=5.5,
                        #            guidance_scale=0,num_inference_steps=50
                        #            ).images[0]
                        pred_img=torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1)
                
                tar_img=PIL.Image.open( base_path/'data'/'images'/ f"{sample['target_name'][0]}.jpg")        
                tar_img= tar_img.resize(img_size, resample=PIL.Image.Resampling.LANCZOS)
                tar_img=torch.from_numpy(np.array(tar_img).astype(np.float32) / 255.0).permute(2,0,1)
                ref_img=PIL.Image.open( base_path/'data'/'images'/ f"{sample['ref_name'][0]}.jpg")        
                ref_img= ref_img.resize(img_size, resample=PIL.Image.Resampling.LANCZOS)
                ref_img=torch.from_numpy(np.array(ref_img).astype(np.float32) / 255.0).permute(2,0,1)
                
                if iterable ==True:
                        grid=tvu.make_grid(torch.stack([ref_img,piclist[0],piclist[1],tar_img],dim=0),nrow=4,padding=0)
                else:
                        grid=tvu.make_grid(torch.stack([ref_img,pred_img,tar_img],dim=0),nrow=3,padding=0)
                if not os.path.exists(ouput_dir):
                        os.mkdir(ouput_dir)
                tvu.save_image(grid,f'{ouput_dir}/{sample["file_name"][0]}.jpg')




def evaluation_folder(folder="evaluation",img_len=20,resize=(256,256),device='cuda'):
        def dummy(images, **kwargs):
                return images, False   
        if not os.path.exists(folder):
                os.mkdir(folder)
                os.mkdir(f"{folder}/reference")
                os.mkdir(f"{folder}/target")
                os.mkdir(f'{folder}/sd1v5')
                triplets=[]
                for dress_type in ['dress','toptee','shirt']:
                        with open(f'data/captions/cap.{dress_type}.val1.json') as f:
                                triplets.extend(json.load(f))
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                        use_auth_token='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',
                        revision="fp16", torch_dtype=torch.float16
                        ).to(device) 
                pipe.safety_checker=dummy
                for i in np.random.permutation(np.arange(0,img_len)):
                        ref_path=triplets[i]["candidate"]
                        target_path=triplets[i]["target"]
                        caption=triplets[i]["captions"]
                        caption=f'{caption[0]} . {caption[1]}'
                        tar_img=PIL.Image.open(f'data/images/{target_path}.jpg')
                        ref_img=PIL.Image.open(f'data/images/{ref_path}.jpg')
                        if resize:
                                tar_img=tar_img.resize(resize, resample=PIL.Image.Resampling.LANCZOS)
                                ref_img=ref_img.resize(resize, resample=PIL.Image.Resampling.LANCZOS)
                        tar_img.save(f'{folder}/target/{caption.replace(" ","_")}.jpg')
                        ref_img.save(f'{folder}/reference/{caption.replace(" ","_")}.jpg')
                        img = pipe(caption,init_image=ref_img).images[0]
                        img.save(f'{folder}/sd1v5/{caption.replace(" ","_")}.jpg')
                res=calculate_fid_given_paths(
                    batch_size=img_len,
                    paths=[f'{folder}/sd1v5',f'{folder}/target'],
                    device=device,
                    dims=2048)
                elog={}
                elog['model']='sd1v5'
                elog['batch_size']=img_len
                elog['dims']=2048
                elog['result']=res
                model_log(f'{folder}/elog',elog)



def evaluation(model_path,output_path,evaluation_path='evaluation',device='cuda'):
        # pipe=FashionImg2ImgPipeline.from_pretrained(model_path,
        #  #       scheduler=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", steps_offset=1, clip_sample=False)
        #                                             ).to(device)
        pipe = FashionImg2ImgPipeline(
            text_encoder=CLIPTextModel.from_pretrained(model_path,subfolder='text_encoder'),
            vae=AutoencoderKL.from_pretrained(model_path,subfolder='vae'),
            unet=UNet2DConditionModel.from_pretrained(model_path,subfolder='unet'),
            tokenizer=CLIPTokenizer.from_pretrained(model_path,subfolder='tokenizer'),
            scheduler=DDPMScheduler.from_config(model_path, subfolder="scheduler"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        ).to('cuda')
        pipe=FashionImg2ImgPipeline.from_pretrained(model_path).to(device)      
                
        if not os.path.exists(f'{evaluation_path}/{output_path}'):
                os.mkdir(f'{evaluation_path}/{output_path}')
        def dummy(images, **kwargs):
                return images, False
        pipe.safety_checker=dummy
        for root,dirs,files in os.walk(f'{evaluation_path}/reference'):
                img_len=len(files)
                print(f'batch size {img_len}')
                for file in files:
                        image = PIL.Image.open(f'{root}/{file}')
                        image=image.resize((512,512))
                        caption=file.replace("_"," ")
                        # img=pipe(caption,init_image=image,
                        #         # guidance_si=3.5, guidance_st=5.5,
                        #         num_inference_steps=50).images[0]
                        #-------------------------#
                        img = pipe.mycall(prompt=caption,
                                init_image=image,
                                num_inference_steps=50,
                                out_dim=256,
                                init_overwrite=False
                        ).images[0]
                        
                        
                        img.save(f'{evaluation_path}/{output_path}/{file}')
        res=calculate_fid_given_paths(
                batch_size=img_len,
                paths=[f'{evaluation_path}/{output_path}',f'{evaluation_path}/target'],
                device=device,
                dims=2048)
        elog={}
        elog['model']=output_path
        elog['model_from']=model_path
        elog['result']=res
        elog['batch_size']=img_len
        elog['dims']=2048

        model_log(f'{evaluation_path}/elog',elog)
        
           
if __name__ =='__main__':
        #evaluation_folder(folder='evaluation_512',resize=(512,512))
        #evaluation("model/attn_vd_ne_32_5e-05_20",'vd','evaluation_512')
        #evaluation("model/info_nce_128_1e-06_20",'full40_lpips','evaluation','cuda:3')
        
        print('??')
        
        # pipe=DDIMPipeline(unet=UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5',subfolder='unet'),
        #                                     scheduler= DDIMScheduler())
        # imgs=pipe()
        # caption='a woman in a purple dress poses for a picture, pinterest contest winner, american barbizon school, product advertising, inverted triangle body type, draped drapes, 2 0 0 8, casual clothing style, official product photo, sexy red dress, oganic rippling spirals, earth tone colors, retaildesignblog.net, large breasts size, anthro'
        # caption='a dress on a mannequin mannequin mannequin mannequin mannequin mannequin mannequin mannequin, the dress\'s lower, ivy vine leaf and flower top, elegant tropical prints, dress of leaves, short flat hourglass slim figure, airbrush dark dress, floral art novuea dress, vine dress, black tunic, feminine girly dress, seductive camisole, flower butterfly vest, feminine slim figure'
        # caption='A black shirt with a skull on it, skull on the chest, badass clothing, skull image on the vest, skull bones, monstrous skull, skull design for a rock band, death skull, breaded skull, aztec skull, of spiked gears of war skulls, wildlife, single aztec skull, with an animal skull for a head, fantasy skull, full skull shaped face cover'
        # pipe =StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5').to('cuda')
        # img=pipe(caption).images[0]
        
        # img.save(f'sample_3.jpg')
        # evaluation_path='evaluation_512'
        # output_path='reference'
        # device='cuda'
        # img_len=20
        # res=calculate_fid_given_paths(
        #         batch_size=img_len,
        #         paths=[f'{evaluation_path}/{output_path}',f'{evaluation_path}/target'],
        #         device=device,
        #         dims=2048)
        # elog={}
        # elog['model']=output_path
        # elog['model_from']='origin'
        # elog['result']=res
        # elog['batch_size']=img_len
        # elog['dims']=2048

        # model_log(f'{evaluation_path}/elog',elog)