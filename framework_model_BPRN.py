# -*- coding:utf-8 -*-
import torch

class BPRN(torch.nn.Module):
    def __init__(self, model):
        super(BPRN, self).__init__()
        self.model = model

    def init_weight(self):
        self.model.init_weight()
        
    def save_parameters(self, path):
        self.model.save_parameters(path)

    def load_parameters(self, fpath):
        return self.model.load_parameters(fpath)

    def init_pretrained_embedding(self, item_embedding):
        self.model.init_pretrained_embedding(item_embedding)

    def forward(self, batch_data):            
        #['user', 'item_lst', 'length of item lst', 'past items', 'length of past items']                         
        # compute score
        # in: (batch, ndoc, hidden) (batch, ndoc, hidden)
        # out: (batch, ndoc)
        scores = self.model.predict(batch_data)
        
        # loop for M positive training
        # first M exs are positive, the rest N-M exs are negative
        # socre difference between a training ex and a negative ex, i.e., s(+) - s(-)                
        score_diffs_lst = []
        for i in range(self.model.numpositives):
            # in: (batch), (batch, negative)
            # out: (batch, negative)
            score_diffs_ = scores[:,i].unsqueeze(1) - scores[:,-self.model.numnegatives:]
            score_diffs_lst.append(score_diffs_)
        score_diffs = torch.cat(score_diffs_lst)
        return -torch.sum(torch.log(torch.sigmoid(score_diffs)))

    def predict(self, batch_data):
        return self.model.predict(batch_data)

    def regularization(self):
        return self.model.regularization()        