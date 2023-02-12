from data_utils import FashionIQDataset
# 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

import argparse
import itertools
import math
import os
import random
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
#from info_nce import InfoNce

import PIL
from accelerate import Accelerator

from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline#, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from huggingface_hub import Repository
from torch.utils.data import random_split

from func_utils import parse_args,get_full_repo_name,freeze_params,unfreeze_params,model_log      

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel
#from lib_vd import UNet2DConditionModel
from diffusers import UNet2DConditionModel
import torchvision.transforms as T



if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


def main():
    args = parse_args()
    args.output_dir=f'{args.output_dir}_{args.train_batch_size*args.gradient_accumulation_steps}_{args.learning_rate}_{args.num_train_epochs}'
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    
    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    if args.unet_path:
        unet== UNet2DConditionModel.from_pretrained(args.unet_path)
    else:
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder='unet')
    if args.CLIPimg_path:
        visionModel = CLIPVisionModel.from_pretrained(args.CLIPimg_path)
        processor = CLIPFeatureExtractor.from_pretrained(args.CLIPimg_path)
    elif os.path.exists(f'{args.pretrained_model_name_or_path}/clip-vit-base-patch32'):
        visionModel=torch.load(f'{args.pretrained_model_name_or_path}/clip-vit-base-patch32')
        #visionModel=CLIPVisionModel.from_pretrained(f'{args.pretrained_model_name_or_path}/clip-vit-base-patch32')# int 0 -255
        processor=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
    else:
        visionModel=CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')# int 0 -255
        processor=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
    
    print('-----------------------------model _loaded -----------------------')
    text_encoder.resize_token_embeddings(len(tokenizer))
    model_dict={}
    model_dict['pretrained']=args.pretrained_model_name_or_path
    model_dict['model_output']=args.output_dir
    model_dict['lr']=args.learning_rate
    model_dict['batch_size']=args.train_batch_size*args.gradient_accumulation_steps
    model_dict['epoch']=args.num_train_epochs

    # freeze params    
    freeze_params(vae.parameters())
    freeze_params(text_encoder.text_model.parameters())
    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # for name,layer in unet.named_parameters():
    #     if 'attentions' in name:
    #         layer.requires_grad=True
    #         if 'attentions2' in name:
    #             torch.nn.init.zeros_(layer)
    #     else :
    #         layer.requires_grad =False
    # train_params=filter(lambda p: p.requires_grad,unet.parameters())
    
    optimizer = torch.optim.AdamW(
        [{  'params':unet.parameters(),
            'lr':args.learning_rate,
            'betas':(args.adam_beta1, args.adam_beta2),
            'weight_decay':args.adam_weight_decay,
            'eps':args.adam_epsilon},
        {
            'params':visionModel.parameters(),
            'lr':1e-4,
            'betas':(args.adam_beta1, args.adam_beta2),
            'weight_decay':args.adam_weight_decay,
            'eps':args.adam_epsilon
        }
        ]
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    train_dataset = FashionIQDataset(
        split='train',
        dress_types=['dress','shirt','toptee'],
        tokenizer=tokenizer,
        preprossor=processor,
        dim=args.resolution
    )
    torch.manual_seed(0)
    #_,train_dataset=random_split(train_dataset,[len(train_dataset)-49,49])
    print(f'training dataset size ={train_dataset.__len__()}, batchsize {args.train_batch_size}')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, \
                                                   drop_last=True,shuffle=True,num_workers=4)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    #args.max_train_steps=train_dataset.__len__()
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    vae,unet,text_encoder, optimizer, train_dataloader, lr_scheduler,visionModel,processor= accelerator.prepare(
     vae,unet, text_encoder, optimizer, train_dataloader, lr_scheduler,visionModel,processor
    )

    # Move vae and unet to device
    text_encoder.text_model.to(accelerator.device)
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    visionModel.to(accelerator.device)
    
    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    text_encoder.text_model.eval()
    #visionModel.eval()
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("pix2pix", config=vars(args))

    # Train! 1*
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    print(f'total_batch_size is {total_batch_size}')
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    zerotoken=tokenizer(
        "",
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    ).input_ids[0]
    batch_size=args.train_batch_size
    zerotoken=zerotoken.to(accelerator.device)
    epoch_loss=[0] 
    # ngau=preprosser(PIL.Image.fromarray((255*torch.randn((256,256,3))+127).clamp(0,255).numpy().astype(np.uint8)))
    # img_null=torch.cat([ngau]*args.train_batch_size,dim=0).to(accelerator.device)
    # print(f'null img shape {img_null.shape}')
    #zeroimage=precessor()
    for epoch in range(args.num_train_epochs):
        print(f'epoch {epoch}-------------')
        step_loss=[]
        unet.train()
        visionModel.train()
        train_loss=0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([unet,visionModel]):
                # Convert images to latent space
                neglect=np.random.randint(0,100)
                if neglect <15:# mute txt & using ref as gt
                    target_latents=vae.encode(batch['ref']).latent_dist.sample().detach()*0.18215
                    tres=text_encoder(torch.cat([zerotoken.unsqueeze(0)]*batch_size,dim=0))
                    text_guidance=tres[0]
                    t_m=tres[1]
                    #print(f' text guidance befor{text_guidance.shape}')
                else:
                    target_latents=vae.encode(batch['target']).latent_dist.sample().detach()*0.18215
                    tres=text_encoder(batch['caption'])
                    text_guidance =tres[0]
                    t_m=tres[1]

                vres=visionModel(pixel_values=batch['ref_i'])
                image_guidance=vres[0]
                i_m=vres[1]

                
               #print(f'img guidance shape{image_guidance.shape},txt shape {text_guidance.shape}')
                cross_guidance=torch.cat([text_guidance,image_guidance],dim=1)
                # Sample noise that we'll add to the latents
                noise = torch.randn(target_latents.shape).to(target_latents.device)
                bsz = target_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device
                ).long()
                
                target_latent_zt= noise_scheduler.add_noise(target_latents,noise,timesteps)

                noise_pred=unet(target_latent_zt,timesteps,encoder_hidden_states=cross_guidance).sample

                loss1 = F.mse_loss(noise_pred,noise,reduction='none').mean([1,2,3]).mean()
                loss2 = F.cosine_similarity(i_m,t_m).mean()*0.5

                loss=loss1+loss2+0.5

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                step_loss.append(float(loss))
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(visionModel.parameters(), 1)
                    accelerator.clip_grad_norm_(unet.parameters(), 1)
                #ema.step(ema_unet,unet)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss=0
                # if global_step % args.save_steps == 0:
                #     save_progress(text_encoder, placeholder_token_id, accelerator, args)
                
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        epoch_loss.append(np.array(step_loss).mean())
        print(f'loss {np.array(step_loss).mean()}')
        accelerator.wait_for_everyone()
        # if np.array(step_loss).mean() >1.5:
        #     break
    model_dict['ave_loss']=np.array(epoch_loss).mean()
    model_dict['epoch_loss']=np.array(epoch_loss).tolist()
    # Create the pipeline using using the trained modules and save it.
    
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline(
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
            scheduler=PNDMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler"),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        pipeline.save_pretrained(args.output_dir)
        torch.save(accelerator.unwrap_model(visionModel),f'{args.output_dir}/clip-vit-base-patch32')
        model_log('training_log',model_dict)
        # Also save the newly trained embeddings
        # save_progress(text_encoder, placeholder_token_id, accelerator, args)
        # torch.save(cat_layer,'model/cat_cov_v3.pth')
        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()



if __name__ == "__main__":
    main()


