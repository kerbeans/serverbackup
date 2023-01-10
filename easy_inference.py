


import  torchvision.utils as  tvu
import torch 
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from data_utils import FashionIQDataset 
from torch.utils.data import DataLoader 
MODLE_PATH ='model/unfreeze_sp20000'

def main():
    pass
    fashioniq= FashionIQDataset('val',['dress','shirt','toptee'],tokenizer=CLIPTokenizer,display=True)
    _,fashioniq=torch.utils.data.random_split(fashioniq,[len(fashioniq)-10,10])

    for sample in fashioniq:
        pass
    


if __name__ =='__main__':
    main()