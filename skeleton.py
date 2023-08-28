from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision
import os
import numpy as np
if __name__ =='__main__':
    # pipe=StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
    # cvm=CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
    # pipe.save_pretrained('model/sd1v5')
    # cvm.save_pretrained('model/sd1v5/clip-vit-base-patch32')
    
    img=[]
    for root,folder,files in os.walk('data/pose'):
        img=[i for i in files if '.jpg' in i]
    out_dir='pose_output'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    img=[img[i] for i in np.random.randint(0,70000,10)]
    for im in img:
        img1=read_image(f'data/images/{im}')
        img2=read_image(f'data/pose/{im}')
        grid=make_grid([img1,img2],nrow=2)
        grid=torchvision.transforms.ToPILImage()(grid)
        grid.save(f'{out_dir}/{im}')