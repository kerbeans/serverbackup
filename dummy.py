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


    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: Union[torch.FloatTensor, PIL.Image.Image],

        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        guidance_si:float =7.5,
        guidance_st:float =2,
        init_overwrite:bool =False,

        **kwargs,
    ):

        self.check_inputs(prompt, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        ## ----------------pixel2pixel------
        text_embeddings=torch.stack([text_embeddings[0],text_embeddings[0],text_embeddings[1]])
        
        # --------------------else 
        #text_embeddings=torch.stack([text_embeddings[0],text_embeddings[1]])
        
        # 4. Preprocess image
        if isinstance(init_image, PIL.Image.Image):
            init_image = preprocess(init_image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            init_image, latent_timestep, batch_size, num_images_per_prompt, text_embeddings.dtype, device, generator
        )
        # print(latents.shape, 'after initnoise',init_image.shape)
        #self.concate_layer.to(device)
        gau_noise=torch.randn(latents.shape,device=device)
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        do_classifier_free_guidance =True
        # 8. Denoising loop
        
        #------------pixel to pixel 
        ref_init = init_image.to(device=device, dtype=text_embeddings.dtype)
        ref_init_dist = self.vae.encode(ref_init).latent_dist
        ref_init = ref_init_dist.sample(generator=generator)
        ref_init = 0.18215 * ref_init
        #print(f' before adding ref{ref_init.shape} {latents.shape}')
        
        if init_overwrite == True:
            latents=gau_noise
        
        #latents=latents*self.scheduler.init_noise_sigma
        for i, t in enumerate(self.progress_bar(timesteps)):
            
            latents = self.scheduler.scale_model_input(latents, t)
            latent_input_1=torch.cat([latents,gau_noise],dim=1)
            latent_input_2=torch.cat([latents,ref_init],dim=1)
            latent_model_input= torch.cat([latent_input_1,latent_input_2,latent_input_2])
            
            # # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # # perform guidance
            
            noise_pred_uncond, noise_pred_img,noise_pred_text = noise_pred.chunk(3)
            # print(f'noise_shape {noise_pred_uncond.shape}')
            
            noise_pred = noise_pred_uncond + guidance_si * (noise_pred_img - noise_pred_uncond)+\
            guidance_st*(noise_pred_text-noise_pred_img)
            # noise_pred=noise_pred_img
                
            #--------------------------oringinal--------------------

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 9. Post-processing
        image = self.decode_latents(latents)

        # 10. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 11. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    







