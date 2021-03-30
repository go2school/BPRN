# -*- coding:utf-8 -*-
import torch
from framework_model_NAISAttention import NAISAttention
from framework_model_FISM import FISM

class NAIS(FISM):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.beta = kwargs["beta_"]
        # attention lay as TKDE 2018
        self.attention_layer = NAISAttention(self.n_factors, self.device, self.beta, self.lambda_, 0)
        
    def init_weight(self):      
        super().init_weight()
        self.attention_layer.init_weight()

    def save_parameters(self, path):
        super().save_parameters(path)
        self.attention_layer.save_parameters(path + '.attention')

    def load_parameters(self, fpath):
        tmp_parameters = torch.load(fpath)
        tmp_attention_parameters = torch.load(fpath + '.attention')
        if tmp_parameters['name'] == 'FISM' and tmp_attention_parameters['name'] == 'NAISAttention':
            if super().load_parameters(fpath) == False:
                print('loading fail FISM', tmp_parameters['name'])
                return False
            if self.attention_layer.load_parameters(fpath + '.attention'):
                print('loading successful attention_layer', tmp_parameters['name'])
                return True
            else:
                print('loading fail attention_layer', tmp_parameters.__class__)
                return False
        else:
            print('loading fail name not match', tmp_parameters.__class__)
            return False
        
    def predict(self, batch_data):
        #['user', 'item_lst', 'length of item lst', 'past items', 'length of past items']
        item_lst_lst = batch_data['item_lst']
        batch, n_docs = item_lst_lst.shape
        past_item_lst = batch_data['past items']
        len_past_item_lst = batch_data['length of past items']

        #embedding on item lst
        # in: (batch, ndoc)
        # out: (batch, ndoc, hidden)
        item_lst_emb = self.item(item_lst_lst)
        _, _, hidden = item_lst_emb.shape
        # in: (batch, ndoc)
        # out: (batch, ndoc)
        item_lst_lst_bais = self.item_bias(item_lst_lst).squeeze(2)
        # add a mask on embedding to remove padding zeros
        # In: (batch, ndoc)
        # Out: (batch, ndoc)
        item_lst_lst_mask = item_lst_lst > 0
        # In: (batch, ndoc)
        # Out: (batch, ndoc, 1)
        item_lst_lst_mask_ = item_lst_lst_mask.unsqueeze(2).float()
        # in: (batch, ndoc, hidden)
        # out: (batch, ndoc, hidden)
        item_lst_emb_ = item_lst_emb * item_lst_lst_mask_
                
        # do embedding on past items
        # in: (batch, sequence)
        # out: (batch, sequence, hidden)
        past_item_emb = self.past_item(past_item_lst)
        # add a mask on embedding to remove padding zeros
        # In: (batch, sequence)
        # Out: (batch, sequence)
        past_item_mask = (past_item_lst > 0).float()
        # In: (batch, sequence)
        # Out: (batch, sequence, 1)
        past_item_mask_ = past_item_mask.unsqueeze(2)
        # In: (batch, sequence, hidden) and (batch, sequence, 1)
        # Out: (batch, sequence, hidden)
        past_item_emb_ = past_item_emb * past_item_mask_
                
        # do attention
        # out: a list of (batch, hidden)
        output_lst = []
        for i in range(n_docs):
            # in: (batch, hidden) (batch, sequence, hidden) (batch, sequence)
            # out: (batch, hidden)
            output, _ = self.attention_layer(item_lst_emb_[:, i, :], past_item_emb_, past_item_mask)
            output_lst.append(output)
        # in: a list of (batch, hidden)
        # out: (batch * ndocs, hidden)
        output_lst_ = torch.cat(output_lst, 0)            
        # in: (batch * ndocs, hidden)
        # out: (ndocs, batch, hidden)
        output_lst_ = output_lst_.view(-1, batch, hidden)
        # in:  (ndocs, batch, hidden)
        # out: (batch, ndocs, hidden)
        profile_rep = output_lst_.permute(1, 0, 2)

        # coefficient on past items 
        # In: (batch, 1)
        # Out: (batch, 1, 1)   
        coeff = len_past_item_lst.pow(-self.alpha)
        coeff_ = coeff.unsqueeze(1).unsqueeze(2)

        # profile
        # in: (batch, ndocs, hidden)
        # out: (batch, ndocs, hidden)
        profile_ = coeff_ * profile_rep

        # compute score
        # in: (batch, ndoc, hidden) (batch, ndoc, hidden)
        # out: (batch, ndoc)
        scores = (profile_ * item_lst_emb_).sum(2) + item_lst_lst_bais

        return scores

    def regularization(self):
        l2_bi_regularization = self.lambda_ * torch.norm(self.item_bias.weight, 2) ** 2
        l2_P_regularization = self.lambda_ * torch.norm(self.past_item.weight, 2) ** 2
        l2_Q_regularization = self.lambda_ * torch.norm(self.item.weight, 2) ** 2                
        l2_attention_regularization = self.attention_layer.regularization(self.lambda_, self.lambda_)
    
        return l2_bi_regularization + l2_P_regularization + l2_Q_regularization + l2_attention_regularization
