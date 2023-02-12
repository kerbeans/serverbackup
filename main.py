
from evaluation.inception import inception_score_eval
from utils.utils import model_log







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
    main()