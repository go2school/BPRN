# -*- coding:utf-8 -*-
import torch

class FISM(torch.nn.Module):
    def __init__(self, kwargs):
        super(FISM, self).__init__()

        #self.n_users = kwargs["n_users"]
        self.n_items = kwargs["n_items"]
        self.n_factors = kwargs["n_factors"]
        self.lambda_ = kwargs["lambda_"]
        self.numnegatives = kwargs["numnegatives"]
        self.numpositives = kwargs["numpositives"]
        self.maxidcg = kwargs["maxidcg"]
        self.device = kwargs["device"]

        # set up alpha
        self.alpha = kwargs["alpha"]           
        # create item bias
        self.item_bias = torch.nn.Embedding(self.n_items, 1)            
	    # create item embeddings
        self.item = torch.nn.Embedding(self.n_items, self.n_factors)
        # create past item embeddings
        self.past_item = torch.nn.Embedding(self.n_items, self.n_factors)
        
    def init_weight(self):
        #init parameters as pytorch does    
        self.item_bias.weight.data.normal_(0, self.lambda_)
        ###self.user.weight.data.normal_(0, 0.05)
        self.item.weight.data.normal_(0, self.lambda_)
        self.past_item.weight.data.normal_(0, self.lambda_)

        # change to xaiver init
        #torch.nn.init.xavier_normal_(self.item_bias.weight)
        #torch.nn.init.xavier_normal_(self.item.weight)
        #torch.nn.init.xavier_normal_(self.past_item.weight)
        
    def save_parameters(self, path):
        pass

    def load_parameters(self, fpath):
        pass

    def init_pretrained_embedding(self, item_embedding):
        self.item.weight.data.copy_(item_embedding)
        self.past_item.weight.data.copy_(item_embedding)
   
    def forward(self, batch_data):            
        # compute score
        # in: (batch, ndoc, hidden) (batch, ndoc, hidden)
        # out: (batch, ndoc)
        scores = self.predict(batch_data)
        
        return scores

    def predict(self, batch_data):
        #['user', 'item_lst', 'length of item lst', 'past items', 'length of past items']
        item_lst_lst = batch_data['item_lst']
        _, n_docs = item_lst_lst.shape
        past_item_lst = batch_data['past items']
        len_past_item_lst = batch_data['length of past items']

        #embedding on item lst
        # in: (batch, ndoc)
        # out: (batch, ndoc, hidden)
        item_lst_emb = self.item(item_lst_lst)
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
        #past_item_emb = self.item(past_item_lst)
        # add a mask on embedding to remove padding zeros
        # In: (batch, sequence)
        # Out: (batch, sequence)
        past_item_mask = past_item_lst > 0
        # In: (batch, sequence)
        # Out: (batch, sequence, 1)
        past_item_mask_ = past_item_mask.unsqueeze(2).float()
        # In: (batch, sequence, hidden) and (batch, sequence, 1)
        # Out: (batch, sequence, hidden)
        past_item_emb_ = past_item_emb * past_item_mask_
        
        # coefficient on past items 
        # In: (batch, 1)
        # Out: (batch, 1)   
        coeff = len_past_item_lst.pow(-self.alpha)
        coeff_ = coeff.unsqueeze(1)

        # profile
        # in: (batch, sequence, hidden)
        # out: (batch, hidden)
        profile_ = coeff_ * (past_item_emb_.sum(1))
        
        # change shape
        # in: (batch, hidden)
        # out: (batch, 1, hidden)
        profile_rep = profile_.unsqueeze(1)
                
        # compute score
        # in: (batch, 1, hidden) (batch, ndoc, hidden)
        # out: (batch, ndoc)
        scores = (profile_rep * item_lst_emb_).sum(2) + item_lst_lst_bais
        
        return scores 

    def regularization(self):
        l2_item_bias_regularization = self.lambda_ * torch.norm(self.item_bias.weight, 2) ** 2
        l2_item_regularization = self.lambda_ * torch.norm(self.item.weight, 2) ** 2  
        l2_past_item_regularization = self.lambda_ * torch.norm(self.past_item.weight, 2) ** 2         
        return l2_item_bias_regularization + l2_item_regularization + l2_past_item_regularization
