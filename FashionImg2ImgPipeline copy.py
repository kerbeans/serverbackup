from diffusers import DiffusionPipeline
import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import os
import PIL.Image
#from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    # EulerAncestralDiscreteScheduler,
    # EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import logging, BaseOutput #, deprecate
from diffusers.pipelines import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker,StableDiffusionPipelineOutput
#from lib_vd import UNet2DConditionModel
from diffusers import UNet2DConditionModel
logger = logging.get_logger(__name__) 
MODEL_PATH='model/info_nce_dc_128_1e-06_40'#'model/info_unet++_64_1e-06_20'#'model/info_nce_128_1e-06_0'
is_accelerate_available=True
PIX2PIX=False

def preprocess(image,dim=224):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((dim, dim), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class FashionImg2ImgPipeline(StableDiffusionImg2ImgPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        requires_safety_checker:bool =True
        super(FashionImg2ImgPipeline,self).__init__()
        #     vae=vae,
        #     text_encoder=text_encoder,
        #     tokenizer=tokenizer,
        #     unet=unet,
        #     scheduler=scheduler,
        #     safety_checker=safety_checker,
        #     feature_extractor=feature_extractor,
        # )
        if not PIX2PIX:
            print('clip imaged model loading')
            if os.path.exists(f"{MODEL_PATH}/clip-vit-base-patch32.pth"):
                self.visionModel=torch.load(f"{MODEL_PATH}/clip-vit-base-patch32.pth")
                print('finished--------')
            else :
                self.visionModel=CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
            self.preprocessor=CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
            self.visionModel.eval()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            #deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            #deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            #deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def prepare_latents(self, init_image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        init_image = init_image.to(device=device, dtype=dtype)
        init_latent_dist = self.vae.encode(init_image).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`init_image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many init images as text prompts to suppress this warning."
            )
            #deprecate("len(prompt) != len(init_image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt * num_images_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `init_image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)

        # get latents
        # print(f' prepare {init_latents.shape},{noise.shape},{type(timestep)}')
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents
        #print(f' end latents shape {init_latents.shape}')
        return latents



    @torch.no_grad()
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
        init_overwrite:bool =True,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 1. Check inputs
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
        #text_embeddings=torch.stack([text_embeddings[0],text_embeddings[0],text_embeddings[1]])
        
        # --------------------else 
        text_embeddings=torch.stack([text_embeddings[0],text_embeddings[1]])
        
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
        
        if init_overwrite == False:
            latents=gau_noise
        
        #latents=latents*self.scheduler.init_noise_sigma
        for i, t in enumerate(self.progress_bar(timesteps)):
            
            # latents = self.scheduler.scale_model_input(latents, t)
            # latent_input_1=torch.cat([latents,gau_noise],dim=1)
            # latent_input_2=torch.cat([latents,ref_init],dim=1)
            # latent_model_input= torch.cat([latent_input_1,latent_input_2,latent_input_2])
            
            # # # predict the noise residual
            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # # # perform guidance
            
            # noise_pred_uncond, noise_pred_img,noise_pred_text = noise_pred.chunk(3)
            # # print(f'noise_shape {noise_pred_uncond.shape}')
            
            # noise_pred = noise_pred_uncond + guidance_si * (noise_pred_img - noise_pred_uncond)+\
            # guidance_st*(noise_pred_text-noise_pred_img)
            # #noise_pred=noise_pred_text
                
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
    
    
    
    @torch.no_grad()
    def mycall(
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
        # guidance_img= Union[torch.FloatTensor, PIL.Image.Image,None] =None,
        init_overwrite:bool =True,
        out_dim=224,
        init_image_i:Union[torch.FloatTensor,PIL.Image.Image]=None,
        **kwargs,
    ):
        # 1. Check inputs
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
        
        # 4. Preprocess image
        if isinstance(init_image, PIL.Image.Image):
            init_image1 = preprocess(init_image,out_dim)
        else:
            init_image1=init_image
        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)#,device=device)
        timesteps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            init_image1, latent_timestep, batch_size, num_images_per_prompt, text_embeddings.dtype, device, generator
        )
        # print(latents.shape, 'after initnoise',init_image.shape)
        #self.concate_layer.to(device)
        gau_noise=torch.randn((latents.shape[0],latents.shape[1],out_dim//8,out_dim//8),device=device)
        #print(f'gau_noise shape {gau_noise.shape}')
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        self.visionModel.to(device)
        if init_overwrite:
            latents=gau_noise 
        if not init_image_i == None:
            if isinstance(init_image_i,torch.FloatTensor):
                init_image2=init_image_i
        elif isinstance(init_image, PIL.Image.Image):
            init_image2 = self.preprocessor(init_image)['pixel_values']
            init_image2=torch.Tensor(init_image2).to(device)
        else:
            init_image2=init_image1.to(device)
        #print(f'init img shape{init_image2.shape}')
        ref_guidance=self.visionModel(pixel_values=init_image2)['last_hidden_state'].to(device)
        
        #print(text_embeddings.shape,ref_guidance.shape)
        if do_classifier_free_guidance:
            #bz*2,77,768 ,bz,50,768
            cross_guidance=torch.cat([ref_guidance]*2,dim=0)
            cross_guidance=torch.cat([text_embeddings,cross_guidance],dim=1)
            
        else:
            cross_guidance=torch.cat([text_embeddings,ref_guidance],dim=1)
        # latents=latents*self.scheduler.init_noise_sigma
        for i, t in enumerate(self.progress_bar(timesteps)):
            
            # latents = self.scheduler.scale_model_input(latents, t)
            
            # #print(f'i {i}, t {t}, lt {latents.shape}, Ct {text_embeddings.shape}, Ci {ref_guidance.shape}')
            if do_classifier_free_guidance:

                latent_model_input= torch.cat([latents]*2,dim=0)
                latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)
                #print(f'latent_input {latent_model_input.shape}, cross {cross_guidance.shape}, t {t.shape}')
                noise_pred=self.unet(latent_model_input,t,encoder_hidden_states=cross_guidance).sample
                noise_ref,noise_tar=noise_pred.chunk(2)
                noise_pred=noise_ref+7.5*(noise_tar-noise_ref)
                
            else :
              #  print('shape----',latents.shape,ref_guidance.shape,text_embeddings.shape)
                noise_pred=self.unet(latents,t,encoder_hidden_states=cross_guidance).sample

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
    
    
    
    @torch.no_grad()
    def pix2pix(
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
        guidance_st:float =1.5,
        init_overwrite:bool =False,
        **kwargs,
    ):
        
        # 1. Check inputs
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
       
        text_embeddings=torch.stack([ text_embeddings[i%batch_size] if i<batch_size*2 else text_embeddings[i%batch_size+batch_size] for i in range(batch_size*3)])
        
        
        # 4. Preprocess image
        if isinstance(init_image, PIL.Image.Image):
            init_image = preprocess(init_image,512)

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


        # 8. Denoising loop
        
        #------------pixel to pixel 
        ref_init = init_image.to(device=device, dtype=text_embeddings.dtype)
        ref_init_dist = self.vae.encode(ref_init).latent_dist
        ref_init = ref_init_dist.sample(generator=generator)
        ref_init = 0.18215 * ref_init
        #print(f' before adding ref{ref_init.shape} {latents.shape}')
        
        if init_overwrite == True:
            latents=gau_noise
            #print(f"latents shape {latent}")
        print(f"latents shape {latents.shape} text shape {text_embeddings.shape}")
        #latents=latents*self.scheduler.init_noise_sigma
        for i, t in enumerate(self.progress_bar(timesteps)):
            
            latents = self.scheduler.scale_model_input(latents, t)
            latent_input_1=torch.cat([latents,gau_noise],dim=1)
            latent_input_2=torch.cat([latents,ref_init],dim=1)
            latent_model_input= torch.cat([latent_input_1,latent_input_2,latent_input_2])
            
            # # # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # # # perform guidance
            
            noise_pred_uncond, noise_pred_img,noise_pred_text = noise_pred.chunk(3)
            # print(f'noise_shape {noise_pred_uncond.shape}')
            
            noise_pred = noise_pred_uncond + guidance_si * (noise_pred_img - noise_pred_uncond)+\
            guidance_st*(noise_pred_text-noise_pred_img)
            #noise_pred=noise_pred_text

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




