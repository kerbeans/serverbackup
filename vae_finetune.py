from diffusers import AutoencoderKL 
from torch.utils.data import Dataset ,DataLoader
import torch
import numpy as np
import PIL
import torch.nn.functional as F
import os
from tqdm import tqdm
from func_utils import model_log

from accelerate import Accelerator





import torch
import torch.nn as nn

# from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


# class LPIPSWithDiscriminator(nn.Module):
#     def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
#                  disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
#                  perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
#                  disc_loss="hinge"):

#         super().__init__()
#         assert disc_loss in ["hinge", "vanilla"]
#         self.kl_weight = kl_weight
#         self.pixel_weight = pixelloss_weight
#         self.perceptual_loss = LPIPS().eval()
#         self.perceptual_weight = perceptual_weight
#         # output log variance
#         self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

#         self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
#                                                  n_layers=disc_num_layers,
#                                                  use_actnorm=use_actnorm
#                                                  ).apply(weights_init)
#         self.discriminator_iter_start = disc_start
#         self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
#         self.disc_factor = disc_factor
#         self.discriminator_weight = disc_weight
#         self.disc_conditional = disc_conditional

#     def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
#         if last_layer is not None:
#             nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
#             g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
#         else:
#             nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
#             g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

#         d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
#         d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
#         d_weight = d_weight * self.discriminator_weight
#         return d_weight

#     def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
#                 global_step, last_layer=None, cond=None, split="train",
#                 weights=None):
#         rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
#         if self.perceptual_weight > 0:
#             p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
#             rec_loss = rec_loss + self.perceptual_weight * p_loss

#         nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
#         weighted_nll_loss = nll_loss
#         if weights is not None:
#             weighted_nll_loss = weights*nll_loss
#         weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
#         nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
#         kl_loss = posteriors.kl()
#         kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

#         # now the GAN part
#         if optimizer_idx == 0:
#             # generator update
#             if cond is None:
#                 assert not self.disc_conditional
#                 logits_fake = self.discriminator(reconstructions.contiguous())
#             else:
#                 assert self.disc_conditional
#                 logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
#             g_loss = -torch.mean(logits_fake)

#             if self.disc_factor > 0.0:
#                 try:
#                     d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
#                 except RuntimeError:
#                     assert not self.training
#                     d_weight = torch.tensor(0.0)
#             else:
#                 d_weight = torch.tensor(0.0)

#             disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
#             loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

#             log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
#                    "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
#                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
#                    "{}/d_weight".format(split): d_weight.detach(),
#                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
#                    "{}/g_loss".format(split): g_loss.detach().mean(),
#                    }
#             return loss, log

#         if optimizer_idx == 1:
#             # second pass for discriminator update
#             if cond is None:
#                 logits_real = self.discriminator(inputs.contiguous().detach())
#                 logits_fake = self.discriminator(reconstructions.contiguous().detach())
#             else:
#                 logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
#                 logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

#             disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
#             d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

#             log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
#                    "{}/logits_real".format(split): logits_real.detach().mean(),
#                    "{}/logits_fake".format(split): logits_fake.detach().mean()
#                    }
#             return d_loss, log

def preprocess(image,dim=None):
    if dim :
        image = image.resize((dim,dim), resample=PIL.Image.Resampling.LANCZOS)
    else :
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image.squeeze(0))
    return 2.0 * image - 1.0

class Imageset(Dataset):
    def __init__(self,filepath):
        self.img_file=[]
        for root,dirs,file in os.walk(filepath):
            self.img_file.extend([f'{root}/{fil}' for fil in file])
    
    def __getitem__(self, index):
        img=preprocess(PIL.Image.open(self.img_file[index]).convert('RGB'),256)
        return img
    def __len__(self):
        return len(self.img_file)
    
    
    
def finetune(model_path,epoch,batch_size,device,output_path,accumulated=1):
    accelerator=Accelerator()
    device =accelerator.device
    vae=AutoencoderKL.from_pretrained(model_path,subfolder='vae').to(device)
    optimizer=torch.optim.AdamW(
        params=vae.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )
    e={}
    e['pretrained']=model_path
    e['model_output']=output_path
    e['lr']=1e-4
    e['batch_size']=batch_size
    e['epoch']=epoch
    e['loss']=[]
    IQdataset=Imageset('data/images')
    IQLoader=DataLoader(IQdataset,batch_size=batch_size,drop_last=True)
    vae,optimizer,IQLoader = accelerator.prepare(
        vae,optimizer,IQLoader
    )
    
    
    
    #lp_loss= LPIPSWithDiscriminator(10001,disc_factor=0).to(device)
    for step in range(epoch):
        vae.train()
        ave_loss=[]
        loss=0
        for et,imgs in tqdm(enumerate(IQLoader)):
            imgs=imgs.to(device)
            latent=vae.encode(imgs).latent_dist
            trace_back=vae.decode(latent.sample()).sample
            #print(f'{latent.shape} {imgs.shape},{trace_back.shape}')
            loss=F.l1_loss(trace_back,imgs,reduction='none').mean([1,2,3]).mean()/accumulated
            #loss ,_=lp_loss(imgs,trace_back,latent,0,et+step*epoch)#accumulated
            loss=loss/accumulated
            #print(loss)
            ave_loss.append(float(loss.detach()))
            accelerator.backward(loss)
            if (et+1)%accumulated==0:
                # print(loss)
                optimizer.step()
                optimizer.zero_grad()
        ave_loss=np.array(ave_loss).mean()
        e['loss'].append(ave_loss)
        print(f'loss {ave_loss} epoch {step}')
    e['ave_loss']=np.array(e['loss']).mean()
    if accelerator.is_main_process:
        vae=accelerator.unwrap_model(vae)
        vae.save_pretrained(output_path)
        model_log('training_log',e)
    
if __name__ =='__main__':
    mdpt='VAEfinetune50_l1'#
    finetune(model_path=mdpt,epoch=5,batch_size=2,device='cuda:0',output_path='VAEfinetune8_l1',accumulated=16)