from typing import List
from torch.utils.data import Dataset
import json
import abc
class BaseDataset(abc.ABC,Dataset):
    def __init__(self,split: str, dress_types: List[str]) -> None:
        super().__init__()
        self.split=split
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
        
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        return len(self.triplets)