from diffusers import StableDiffusionImg2ImgPipeline



class FashionImg2ImgPiepline(StableDiffusionImg2ImgPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        