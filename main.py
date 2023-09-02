
from evaluation.inception import inception_score_eval
from utils.utils import model_log

from evaluation.evaluate import validation_image
from pipelines.pipeline import FashionImg2ImgPipeline
from dataset.FashionIQDataset import FashionIQDataset

from transformers import CLIPTokenizer,CLIPFeatureExtractor
from utils.utils import get_obj_from_str, instantiate_from_config, create_dataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import argparse

parser =argparse.ArgumentParser()
parser.add_argument('--setting')
arg=parser.parse_args()



def test():
    from diffusers.utils import load_image

    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )

    import cv2
    from PIL import Image
    import numpy as np

    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image

    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    import torch

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    from diffusers import UniPCMultistepScheduler

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid
    
    prompt = ", best quality, extremely detailed"
    prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]
    from pipelines.pipeline import dummy_safety_checker
    # pipe.safety_checker= dummy_safety_checker
    output = pipe(
        prompt,
        canny_image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
        )
    print(type(output.images[0]))
    image_grid(output.images, 2, 2).save('aa.jpg')





def main():
    
    conf =OmegaConf.load('config/training.yaml')
    subConf=conf[arg.setting]
    trainingset,validationset=create_dataset(subConf)
    Trainloader = DataLoader(trainingset)
    for batch in Trainloader:
        print(batch.keys())
        exit()
    
    
    
    
    






if __name__ == '__main__':
    main()
    