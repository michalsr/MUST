from model import clip_classifier 
from unicl.model import build_unicl_model 
import torch.nn as nn 
import torch 
class Ensemble_CLIP(nn.Module):
    def __init__(self,must_args,unicl_args,is_train=False):        
        super().__init__()
        self.must = clip_classifier(must_args)
        self.unicl = build_unicl_model(unicl_args)
    def forward(self,img,txt):
        #unicl 
        features_img, features_text, tau = self.unicl(img,txt)
        unicl_output = tau*features_img@features_text.t()
        #must 
        must_output = self.must(img)
        combined = torch.stack((unicl_output,must_output),dim=0)
        avg_logit = .5*torch.sum(combined,dim=0)
        return avg_logit
