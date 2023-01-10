import torch
import  torch.nn.functional as F
import numpy as np
import tqdm
from diffusers import AutoencoderKL,UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import DataLoader


from data_utils import FashionIQDataset
from func_utils import ArgParse, freeze_params

    

def main():
    args=ArgParse()
    print(args.device)
    if args.from_hub:
        pass    
    elif args.pretrained_dir :
        apath = args.pretrained_dir
        tokenizer=CLIPTokenizer.from_pretrained(apath,subfolder='tokenizer')
        vae=AutoencoderKL.from_pretrained(apath, subfolder='vae')
        text_encoder=CLIPTextModel.from_pretrained(apath,subfolder='text_encoder')
        noise_scheduler = DDPMScheduler.from_config(apath, subfolder="scheduler")
        if args.unet_path == None :
            unet=UNet2DConditionModel.from_pretrained(apath,subfolder='unet')
        else:
            print(f' unet load from {args.unet_path}')
            unet=UNet2DConditionModel.from_pretrained(args.unet_path)
    if 'pix2pix' not in args.pretrained_dir:
    #print(unet)
        conv_in=unet.conv_in.weight
        conv_in=torch.nn.Parameter(torch.cat([conv_in,torch.zeros(conv_in.shape)],dim=1))

        unet.conv_in=torch.nn.Conv2d(8,320,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        unet.conv_in.weight=conv_in
    
    model_dict={}
    model_dict['pretrained']=args.pretrained_dir
    model_dict['model_output']=args.output_dir
    model_dict['lr']=args.learning_rate
    model_dict['batch_size']=args.batch_size
    model_dict['epoch']=args.epoch
    model_dict['accumulation']=args.accumulation_steps
    
    
    
    text_encoder.resize_token_embeddings(len(tokenizer))
    device = args.device
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    
    freeze_params(vae.parameters())
    freeze_params(text_encoder.text_model.parameters())
    
    vae.eval()
    text_encoder.text_model.eval()
    
    train_data=FashionIQDataset(
        split='train',
        dress_types=['dress','shirt','toptee'],
        tokenizer=tokenizer,
        dim=args.resolution
    )
    torch.manual_seed(0)
    train_data ,_ =torch.utils.data.random_split(train_data,[len(train_data)-10,10])
    
    optimizer=torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9,0.99),
        weight_decay=1e-2,
        eps=1e-8
    )
    
    batch_size=args.batch_size
    max_train_steps=batch_size*args.epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps *1,# args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * 1#args.gradient_accumulation_steps,
    )
    zerotoken=tokenizer(
        "",
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    ).input_ids[0]
    zerotoken=torch.stack([zerotoken]*batch_size,dim=0).to(device)
    train_dataloader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,drop_last=True)
    epoch_loss=[]
    for ep in range(args.epoch):
        unet.train()
        step_loss=[]
        for step,batch in enumerate(train_dataloader):
            for i in ['target','caption','ref']:
                batch[i]=batch[i].to(device)
            target_latents=vae.encode(batch['target']).latent_dist.sample().detach()*0.18215
            neglect=np.random.randint(0,100)
            if neglect<5: #mute img
                ref_ci=torch.randn(target_latents.shape).to(target_latents.device)
                encoder_hidden_states = text_encoder(batch["caption"])[0]
            elif neglect <10: # mute text
                ref_ci = vae.encode(batch["ref"]).latent_dist.sample().detach()*0.18215
                encoder_hidden_states=text_encoder(zerotoken)[0]
            elif neglect <15: # mute both
                encoder_hidden_states=text_encoder(zerotoken)[0]
                ref_ci=torch.randn(target_latents.shape).to(target_latents.device)
            else: # unmute
                ref_ci = vae.encode(batch["ref"]).latent_dist.sample().detach()*0.18215
                encoder_hidden_states = text_encoder(batch["caption"])[0]
                
            noise = torch.randn(target_latents.shape).to(target_latents.device)
            bsz = target_latents.shape[0]
                # Sample a random timestep for each image
            timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device
            ).long()
            target_latent_zt= noise_scheduler.add_noise(target_latents,noise,timesteps)

            latents_input=torch.cat([target_latent_zt,ref_ci],dim=1)
            noise_pred=unet(latents_input,timesteps,encoder_hidden_states).sample

            loss = F.mse_loss(noise_pred,noise,reduction='none').mean([1,2,3]).mean()
            step_loss.append(float(loss))
            loss=loss/args.accumulation_steps
            loss.backward()
            if step%args.accumulation_steps==0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        epoch_loss.append(np.array(step_loss).mean())
        print(f'epoch {ep} finished')
    model_dict['ave_loss']=np.array(epoch_loss).mean()
    model_dict['epoch_loss']=np.array(epoch_loss).tolist()
    unet.save_pretrained(args.output_dir+f'/vae_ep{ep}lr{args.learning_rate}')
        






if __name__=='__main__':
    main()