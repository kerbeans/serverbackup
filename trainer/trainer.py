from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel,CLIPFeatureExtractor
from accelerate import Accelerator
from accelerate.utils import set_seed
from info_nce import InfoNCE
from PIL import Image
import numpy as np
import os
import torch
import torch.functional as F
import math
import tqdm
from module.ldm.unet_model import UNet2DConditionModel
from dataset.FashionIQDataset import FashionIQDataset_light, collate_fn_light
from utils.utils import model_log ,freeze_params
import yaml


class base_trainer():
    def __init__(self,yamlpath,setting) -> None:
        with open(yamlpath,'r') as f:
            self.args=yaml.load(f,Loader=yaml.FullLoader)[setting]
        self.load_model()
        self.log={}
        self.log['pretrained']=self.args.pretrained_model_name_or_path
        self.log['model_output']=self.args.output_dir
        self.log['lr']=self.args.learning_rate
        self.log['batch_size']=self.args.train_batch_size*self.args.gradient_accumulation_steps
        self.log['epoch']=self.args.num_train_epochs
        



   
    def train(self):
        
        accelerator = Accelerator(
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                mixed_precision=self.args.mixed_precision)
        if self.args.seed is not None:
            set_seed(self.args.seed)
        
        noise_scheduler = DDPMScheduler.from_config(self.args.pretrained_model_name_or_path, subfolder="scheduler")
        train_dataloader=self.prepare_dataloader()
        #--------------prepare optimizer
        self.prepare_optimizer(len(train_dataloader))
        #---------------prepare InfoNCE loss
        NCE_loss=InfoNCE()
        freeze_params(self.vae.parameters())
        freeze_params(self.visionModel.parameters())
        freeze_params(NCE_loss.parameters())
        self.vae.eval()
        self.visionModel.eval()
        self.unet.train()
        self.text_encoder.train()
        
        
        self.vae.apply(self.inplace_relu)
        self.visionModel.apply(self.inplace_relu)
        self.unet.apply(self.inplace_relu)
        
        self.vae,self.unet,self.text_encoder, self.optimizer, train_dataloader, self.lr_scheduler,self.visionModel,self.processor,NCE_loss= accelerator.prepare(
        self.vae,self.unet, self.text_encoder, self.optimizer, train_dataloader, self.lr_scheduler,self.visionModel,self.processor,NCE_loss
        )
        #-------------------prepare progress bar
        
        total_batch_size = self.args.train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps
        print(f'total_batch_size is {total_batch_size}')
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        
        global_step = 0
        batch_size=self.args.train_batch_size
        epoch_loss=[0]
        #-------------------- prepare zero latent
        zerotoken=self.prepare_zero_latent().to(accelerator.device).detach()
        
        for epoch in range(self.args.num_train_epochs):
            print(f'epoch {epoch}-------------')
            step_loss=[]
            train_loss=0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate([self.unet,self.text_encoder]):
                    ref_img,tar_img,captions=self.prepare_triplets(batch,accelerator.device,methods=[self.preprocess,self.processor])
                    percentage = np.random.randint(0,100)
                    ref_vae,ref_clip=ref_img[0],ref_img[1]['pixel_values'][0]
                    tar_vae,tar_clip=tar_img[0],tar_img[1]['pixel_values'][0]
                    if percentage <10:# 0-10 NCE_loss and CLIP similarity directly on noisy image?
                        
                        
                        
                        pass
                    elif percentage <30:# 10-30 oringal diffusion e(Z,CIref,null) -> I_ref,
                        target_latents=self.vae.encode(ref_vae).latent_dist.sample().detach()*0.18215
                        vres=self.visionModel(pixel_values=ref_clip)
                        image_guidance=vres[0]
                        
                        tres=self.text_encoder(torch.cat([zerotoken.unsqueeze(0)]*batch_size,dim=0))
                        text_guidance=tres[0]
                        
                    else :#   30-100 conditional diffusion e(Z,CIref,CT) ->I_tar , and NCE_loss
                        target_latents=self.vae.encode(tar_vae).latent_dist.sample().detach()*0.18215
                        tres=self.text_encoder(captions)
                        text_guidance =tres[0]
                        t_m=tres[1]
                        
                        vres=self.visionModel(pixel_values=ref_clip)
                        image_guidance=vres[0]
                        i_m=vres[1]
                        
                        tar_i_m=self.visionModel(pixel_values=tar_clip)[1]
                        compose=i_m+t_m
                        nce_loss=NCE_loss(compose,tar_i_m)
                        
                        
                    if percentage>=0:
                            pass

            
            
                    loss1=F.mse_loss()
                    loss=loss1+nce_loss
                    
                    
                    
                    step_loss.append(float(loss.cpu().detach()))
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.visionModel.parameters(), 1)
                        accelerator.clip_grad_norm_(self.unet.parameters(), 1)
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.prepare_optimizeroptimizer.zero_grad()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    train_loss=0
                
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
            
            if epoch % 5 == 0 :
                print(f'saving temp model on epoch {epoch}')
                self.save_progress(self.unet, accelerator)
                self.save_progress(self.text_encoder, accelerator)
            epoch_loss.append(np.array(step_loss).mean())
            print(f'loss {np.array(step_loss).mean()}')
            accelerator.wait_for_everyone()
            self.log['ave_loss']=np.array(epoch_loss).mean()
            self.log['epoch_loss']=np.array(epoch_loss).tolist()
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline(
                text_encoder=accelerator.unwrap_model(self.text_encoder),
                vae=self.vae,
                unet=accelerator.unwrap_model(self.unet),
                tokenizer=self.tokenizer,
                scheduler=DDPMScheduler.from_config(self.args.pretrained_model_name_or_path, subfolder="scheduler"),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
            pipeline.save_pretrained(self.args.output_dir)
            accelerator.unwrap_model(self.visionModel).save_pretrained(f'{self.args.output_dir}/clip-vit-base-patch32')
            model_log('training_log',self.log)
        accelerator.end_training()
        
    def prepare_triplets(self,batch,device,methods):
        captions=batch['caption'].to(device)
        ref_img=[]
        tar_img=[]
        for i in methods:
            ref_img.append(i(batch['ref']).to(device))
            tar_img.append(i(batch['target']).to(device))
        return ref_img,tar_img,captions
        
    def prepare_dataloader(self):
        train_dataset = FashionIQDataset_light(
            split='train',
            dress_types=['dress','shirt','toptee'],
            tokenizer=self.tokenizer,
            dim=self.args.resolution,
        )
        torch.manual_seed(0)
        #_,train_dataset=random_split(train_dataset,[len(train_dataset)-49,49])
        print(f'training dataset size ={len(train_dataset)}, type {train_dataset.__class__.__name__}')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, \
                                                    drop_last=True,shuffle=True,num_workers=8,pin_memory=True,
                                                    collate_fn=collate_fn_light
                                                    )
        return train_dataloader
    
    def prepare_optimizer(self, num_update_steps_per_epoch):   
        with self.args as args:      
            self.optimizer = torch.optim.AdamW(
            [{  'params':self.unet.parameters(),
                'lr':args.learning_rate,
                'betas':(args.adam_beta1, args.adam_beta2),
                'weight_decay':args.adam_weight_decay,
                'eps':args.adam_epsilon},
            {
                'params':self.text_encoder.text_model.parameters(),
                'lr':1e-4,
                'betas':(args.adam_beta1, args.adam_beta2),
                'weight_decay':args.adam_weight_decay,
                'eps':args.adam_epsilon},
            # {
            #     'params':visionModel.parameters(),
            #     'lr':1e-4,
            #     'betas':(args.adam_beta1,args.adam_beta2),
            #     'weight_decay':args.adam_weight_decay,
            #     'eps':args.adam_epsilon}
            ])      
            
            if args.max_train_steps is None:
                args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
                overrode_max_train_steps = True
                
            self.lr_scheduler = get_scheduler(
                                    args.lr_scheduler,
                                    optimizer=self.optimizer,
                                    num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                                    num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
                                )
            num_update_steps_per_epoch = math.ceil(num_update_steps_per_epoch / args.gradient_accumulation_steps)
            if overrode_max_train_steps:
                args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            # Afterwards we recalculate our number of training epochs
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
        
    def prepare_zero_latent(self):
        zerotoken=self.tokenizer(
                    "",
                    padding="max_length",
                    truncation=True,
                    return_tensors='pt'
                ).input_ids[0]
        return zerotoken
        
        
    def load_model(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        if self.args.unet_path:
            self.unet== UNet2DConditionModel.from_pretrained(self.args.unet_path)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(self.pretrained_model_name_or_path,subfolder='unet')
        if self.args.CLIPimg_path:
            self.visionModel = CLIPVisionModel.from_pretrained(self.args.CLIPimg_path)
            self.processor = CLIPFeatureExtractor.from_pretrained(self.args.CLIPimg_path)
        elif os.path.exists(f'{self.args.pretrained_model_name_or_path}/clip-vit-base-patch32'):
            #visionModel=torch.load(f'{args.pretrained_model_name_or_path}/clip-vit-base-patch32')
            self.visionModel=CLIPVisionModel.from_pretrained(f'{self.args.pretrained_model_name_or_path}/clip-vit-base-patch32')# int 0 -255
            self.processor=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
        else:
            self.visionModel=CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')# int 0 -255
            self.processor=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
    
    def inplace_relu(self,m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace=True
    
    def save_progress(self,model,accelerator):
        torch.save(accelerator.unwrap_model(model), os.path.join(f'{self.args.output_dir}',f"temp_{model.__class__.__name__}"))


    def preprocess(self,image,dim=256):
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
    
    def loss_fun():
        pass
    
    def log_images():
        pass
