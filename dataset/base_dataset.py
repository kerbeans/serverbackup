from typing import List
from torch.utils.data import Dataset
import json
import abc
import os
class BaseDataset(abc.ABC,Dataset):
    def __init__(self,split: str, dress_types: List[str]) -> None:
        super().__init__()
        self.split=split
        #self.data_path='../yiyang/data/fashionIQ/image_data'
        self.data_path='data/images'
        self.triplets: List[dict] = []
        if self.split is None:
            return 
        elif split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")
        for dress_type in dress_types:
            with open(f'data/captions/cap.{dress_type}.{split}1.json') as f:
                self.triplets.extend(json.load(f))
                
        #----------------------filtering ----------------------
        # invalid=self.check_exist()
        # new_trip=[]
        # for i in self.triplets:
        #     flag= True
        #     for j in invalid:
        #         if (i['candidate']==j['candidate']) and (i['target']==j['target']):
        #             flag=False
        #             break
        #     if flag :
        #         new_trip.append(i)
                
                    
        # self.triplets=new_trip
        
        
    def check_exist(self):
        print(f'check imgs under path {self.data_path}')
        invalid=[]
        img_under_path=[]
        for subfolder in ['dress','shirt','toptee']:
            temp=[]
            for root,folder,file in os.walk(f"{self.data_path}/{subfolder}"):
                temp=[i.replace('.jpg','') for i in file]
            img_under_path.extend(temp)
            
        print(f"images under path {len(img_under_path)}, inital triplets len {self.__len__()}")
        for i in self.triplets:
            if (i['candidate'] not in img_under_path) or (i['target'] not in img_under_path):
                invalid.append(i)
        return invalid
        
        
        
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        return len(self.triplets)