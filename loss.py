import torch.nn.functional as F
import torch
from info_nce import InfoNCE


def naive_info_nce(anchor,p1,p2,base):
    '''
    intention: loss target->noise(target)+Ci,Ct -> target
    
    '''
    InfoNCE()
    
    pass


def KL_distance(anchor,pi,pt):
    it_mediate=0.5*pt+0.5*pi
    return F.kl_div(it_mediate.log(),anchor,reduction='none').mean([1,2,3]).mean()


def quadruplet_loss(anchor,p1,p2,n,alpha,beta):
    '''
    feature size b,c,h,w: _, 3,32,32 
    '''
    KL_distance(anchor,p1)-KL_distance(anchor,n)+alpha ,    
    
def intermediate_loss(anchor,pi,pt):
    '''
    feature size b,c,h,w: _, 3,32,32 
    anchor pt should be detached 
    Wasserstein
    '''
    IT_mediate=0.5*pi+0.5*pt
    l1= F.mse_loss(anchor,IT_mediate,reduction='none').mean([1,2,3]).mean()
    
    return l1

def restrict_loss(method=None):
    pass






dummy_anchor=torch.ones((1,3,32,32))
dummy_p1=torch.ones((1,3,32,32))
print(F.kl_div(dummy_p1,dummy_anchor,reduction='mean'))