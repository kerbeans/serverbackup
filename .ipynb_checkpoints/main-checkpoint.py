
from evaluation.inception import inception_score_eval
from utils.utils import model_log

from evaluation.evaluate import validation_image





def main():
    path='data/images'
    split=20
    elog={}
    elog['mean'],elog['std']=inception_score_eval(path=path,device='cuda:2',splits=split)
    print(elog)
    elog['path']=path
    elog['split']=split
    elog['comments']='inception score'
    model_log('scripts_public/evaluation_log',elog)

if __name__ == '__main__':
    model_path='model/clip_short_cut_4_1e-06_40' 
    validation_image("short_cut",model_path,batch=16,device='cuda',flag='dc',dim=256,overwrite=False)(flag='multi',image_num=20)