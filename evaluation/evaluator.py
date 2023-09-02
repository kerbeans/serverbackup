import os
from fid_score  import validate_by_fid
from torch.utils.data import DataLoader
from utils.utils import instantiate_from_config 
import shutil

class evaluator():
    def __init__(self,log_path:str,groundTruth_path:str='data',output_path:str='tmp_validation_epoch',device='cuda'):
        
        self.log_path=log_path
        self.logger=[]
        self.device=device
        self.groundTruth_path =groundTruth_path # source image
        self.output_path:output_path    #image generated 
        self.prev=None
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.mkdir(self.output_path)
    
    def gen_images(self,model,params,dataloader,image_num=64,seed=0):  # unfinished    
        for batch in dataloader:
            output = model()#////
            output.save(f"{self.output_path}/{batch[0]}")# img name 
    
    
    
    
    
    
    def _calculate_score(self):
        fid_value,self.prev=validate_by_fid([self.groundTruth_path,self.output_path],
                                       batch_size=8,device=self.device,prev=self.prev)
        return fid_value
    def save_best(self,model,params:dict,model_path:str,dataloader_v:DataLoader,epoch_num,image_num=64,seed=0):
        
        self.gen_images(model,params,dataloader_v,image_num,seed=0)
        fid_value=self._calculate_score()
        self.logger.append({epoch_num:fid_value}) 
        with open(self.log_path,'w') as f:
            f.write(str(self.logger))    
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        if min(self.logger)==fid_value:
            return True 
        else:
            return False
        
    def on_save_checkpoint(self,checkpoint):
    
    
    def on_load_checkpoint(self,)