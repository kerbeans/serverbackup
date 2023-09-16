from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel,CLIPFeatureExtractor, PretrainedConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
# from info_nce import InfoNCE
from PIL import Image
import numpy as np
import os
import torch
import torch.functional as F
import math
import tqdm
from module.ldm.unet_2d_condition import UNet2DConditionModel
from dataset.FashionIQDataset import FashionIQDataset_light, collate_fn_light
from accelerate.utils import ProjectConfiguration, set_seed

from utils.utils import model_log ,freeze_params
import yaml
import transformers
import diffusers
import wandb
import random 

from module.ControlNet import ControlNetModel


class base_trainer():
    def __init__(self,dataloader,conf) -> None:
        self.args=conf['Training']
        self.load_model()
        self.log={}
        self.log['pretrained']=self.args.pretrained_model_name_or_path
        self.log['model_output']=self.args.output_dir
        self.log['lr']=self.args.learning_rate
        self.log['batch_size']=self.args.train_batch_size*self.args.gradient_accumulation_steps
        self.log['epoch']=self.args.num_train_epochs
        
        self.dataloader=dataloader
        
        #---------------------- args undefined --------------------
        output_dir =None
        logging_dir =None# modified 
        self.accelerator_config= ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)# 
        
        
        #-----------------------------------

        

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
        
    '''
    # def prepare_dataloader(self):
    #     train_dataset = FashionIQDataset_light(
    #         split='train',
    #         dress_types=['dress','shirt','toptee'],
    #         tokenizer=self.tokenizer,
    #         dim=self.args.resolution,
    #     )
    #     torch.manual_seed(0)
    #     #_,train_dataset=random_split(train_dataset,[len(train_dataset)-49,49])
    #     print(f'training dataset size ={len(train_dataset)}, type {train_dataset.__class__.__name__}')
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, \
    #                                                 drop_last=True,shuffle=True,num_workers=8,pin_memory=True,
    #                                                 collate_fn=collate_fn_light
    #                                                 )
    #     return train_dataloader
    '''
    
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
    
    def wandb_init(self,):
# start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.args.expsetting,
            
            # track hyperparameters and run metadata
            config={
            "learning_rate":self.args.learning_rate,
            "architecture": None,
            "dataset": "FashionIQ",
            "epochs": self.args.num_train_epochs, # epochs
            }
        )
        self.wandb=wandb
# simulate training
        epochs = self.args.num_train_epochs
        offset = random.random() / 5
        for epoch in range(2, epochs):
            acc = 1 - 2 ** -epoch - random.random() / epoch - offset # 
            loss = 2 ** -epoch + random.random() / epoch + offset #
                
                # log metrics to wandb
            wandb.log({"acc": acc, "loss": loss})
    
# [optional] finish the wandb run, necessary in notebooks
        wandb.finish()
        hasattr(self,wandb)
    
    
    
    
    def _image_grid(self,imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid
    
    def import_model_class_from_model_name_or_path(self,pretrained_model_name_or_path: str, revision: str):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

            return RobertaSeriesModelWithTransformation
        else:
            raise ValueError(f"{model_class} is not supported.")
    
    
    def _train_control_one_epoch(self,epoch,step,batch):
        with self.accelerator.accumulate(self.controlnet):
                # Convert images to latent space
            latents = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
                # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

            controlnet_image = batch["conditioning_pixel_values"].to(dtype=self.weight_dtype)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

            # Predict the noise residual
            model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                ).sample

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = self.controlnet.parameters()
                self.accelerator.clip_grad_norm_(params_to_clip, 1)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)# key arguments

            # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    image_logs = log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        controlnet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        if global_step >= args.max_train_steps:
            break
    
    

    def train_control(self,dataloader):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=self.report_to,
            project_config=self.accelerator_project_config,
        )
        
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        
        
        
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        #--------------------------------args undefined -------------------------------------
        args=None
        text_encoder_cls = self.import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)        
        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        self.controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    
        for epoch in range(0, args.num_train_epochs):
            for step, batch in enumerate(self.dataloader):
                self._train_control_one_epoch(epoch,step,batch) #????

        #--------------------------------args undefined -------------------------------------
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.controlnet = self.accelerator.unwrap_model(self.controlnet)
            self.controlnet.save_pretrained(args.output_dir)


        self.accelerator.end_training()
