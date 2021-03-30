# -*- coding:utf-8 -*-
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import random
import pandas as pd
import collections
import numpy as np
from util import read_int_lines, shuffle_multiple_lst

##################################################################################################
# sampler dataset
##################################################################################################
class UCSampler:
    """
        negative sampling for pointwise and pairwise
    """
    def __init__(self, train_pos_fname, max_batch_num_pos, max_batch_num_items):
        # prepare training set
        self.u_past_pos_ex_lst = self.make_user_past_positive_sequence(train_pos_fname, max_batch_num_pos)        
        # set parameters
        self.max_batch_num_items = max_batch_num_items
        self.max_batch_num_pos = max_batch_num_pos        

    def make_user_past_positive_sequence(self, fname, num_pos):   
        dt = pd.read_csv(fname)
        ex_lst = []
        for tuple_ in dt.itertuples():
            u = tuple_.user
            items = [int(l) for l in tuple_.items.split(' ')]
            past_items = items[:-num_pos]
            pos_items = items[-num_pos:]
            #user,past_items,pos_items
            ex_lst.append((u, past_items, pos_items))
        return ex_lst
    
    def randomly_sample_negative_examples(self, u_past_pos_ex_lst, num_negative, all_items):
        ret_pos_and_neg_ex_lst = []
        for pos_ex in u_past_pos_ex_lst:
            #label,user,target,train_items
            user = pos_ex[0]
            train_items = pos_ex[1]
            pos_items = pos_ex[2]        
            #rest negatives
            rest_negs = all_items - set(train_items) - set(pos_items)
            num_negatives = num_negative if len(rest_negs) > num_negative else len(rest_negs)
            sampled_negs = random.sample(rest_negs, num_negatives)                
            # put neg
            rate_lst = [1 for i in range(len(pos_items))] + [0 for i in range(len(sampled_negs))]
            # append
            ret_pos_and_neg_ex_lst.append((user, pos_items, sampled_negs, train_items.copy(), rate_lst))
        return ret_pos_and_neg_ex_lst

    def randomly_sample_neg_ranking(self, all_items, num_positive, num_negative, isshuffle):
        user_pos_neg_past_rate_lst = self.randomly_sample_negative_examples(self.u_past_pos_ex_lst, num_negative, all_items)
        s = UCDataset()
        s.buildDataset(user_pos_neg_past_rate_lst, self.max_batch_num_items, num_positive, num_negative, isshuffle)
        return s    

##################################################################################################
# dataset
##################################################################################################
class UCBasicDataset(Dataset):
    def __init__(self):
        pass

    def make_dataset(self, user_pos_neg_past_rate_lst, max_batch_num_items, max_batch_num_pos, max_batch_num_neg):
        u_lst = []
        ex_lst_lst = []
        r_lst_lst = []
        rest_j_lst = []
        len_rest_j_lst = []       
        len_ex_lst_lst = []
        ndocs = max_batch_num_pos + max_batch_num_neg
        #user, target+sampled_negs, train_items
        for u, pos_ex_lst, neg_ex_lst, rest_j, r_lst in user_pos_neg_past_rate_lst:
            u_lst.append(u)          
            rest_j = list(rest_j)
            # positive + negative items
            ex_lst = list(pos_ex_lst) + list(neg_ex_lst)
            if len(ex_lst) > ndocs:#cutoff
                ex_lst = ex_lst[-ndocs:]
                len_ex_lst_lst.append(ndocs)  
                r_lst = r_lst[-ndocs:]
            else:                 
                len_ex = len(ex_lst)
                len_ex_lst_lst.append(len_ex)
                ex_lst += (ndocs - len_ex) * [0]
                r_lst += (ndocs - len_ex) * [0]
            ex_lst_lst.append(ex_lst) 
            r_lst_lst.append(r_lst)       
            # hisotry items
            if len(rest_j) > max_batch_num_items:#cutoff
                len_rest_j = max_batch_num_items                    
                rest_j = rest_j[-max_batch_num_items:]                    
            else: 
                len_rest_j = len(rest_j)
                rest_j = (max_batch_num_items - len(rest_j)) * [0] + rest_j
            rest_j_lst.append(rest_j)
            len_rest_j_lst.append(len_rest_j)            
                    
        return u_lst, ex_lst_lst, len_ex_lst_lst, rest_j_lst, len_rest_j_lst, r_lst_lst

class UCDataset(UCBasicDataset):
    def __init__(self):
        super()

    def buildDataset(self, user_posneg_past_rate_lst, max_batch_num_items, max_batch_num_pos, max_batch_num_negs, isshuffle=True):
        #read data
        train_u_lst, train_ex_lst, train_len_ex_lst, train_rest_j_lst, train_len_rest_j_lst, train_r_lst_lst = self.make_dataset(user_posneg_past_rate_lst, max_batch_num_items, max_batch_num_pos, max_batch_num_negs)

        #shuffle data
        if isshuffle == True:
            _, train_u_lst, train_ex_lst, train_len_ex_lst, train_rest_j_lst, train_len_rest_j_lst, train_r_lst_lst = shuffle_multiple_lst(train_u_lst, train_ex_lst, train_len_ex_lst, train_rest_j_lst, train_len_rest_j_lst, train_r_lst_lst)
                
        self.x_user = torch.LongTensor(train_u_lst)
        self.x_ex_lst = torch.LongTensor(train_ex_lst) 
        self.max_batch_num_pos = max_batch_num_pos      
        self.max_batch_num_negs = max_batch_num_negs
        self.x_item_ex_lst_len = torch.FloatTensor(train_len_ex_lst)    
        self.x_rest_items = torch.LongTensor(train_rest_j_lst)        
        self.x_rest_items_len = torch.FloatTensor(train_len_rest_j_lst) 
        self.x_rate_lst = torch.FloatTensor(train_r_lst_lst)     
        self.len = len(train_u_lst)

    def explain_item(self):
        return ['user', 'item_lst', 'length of item lst', 'past items', 'length of past items', 'rate_lst']    

    def __getitem__(self, index):
        return self.x_user[index], self.x_ex_lst[index], self.x_item_ex_lst_len[index], self.x_rest_items[index], self.x_rest_items_len[index], self.x_rate_lst[index]

    def __len__(self):
        return self.len
    
class SeqUCDataset(UCDataset):
    def __init__(self):
        super()

    def make_all_lst(self, fname, num_pos, num_neg):
        ex_lst = []
        dt = pd.read_csv(fname)
        for tuple_ in dt.itertuples():
            #user,items,neg_items
            user = tuple_.user
            items = [int(l) for l in tuple_.items.split(' ')]
            neg_items = [int(l) for l in tuple_.neg_items.split(' ')][:num_neg]
            past_items = items[:-num_pos]
            pos_items = items[-num_pos:]
            rate_lst = [1 for i in range(num_pos)] + [0 for i in range(len(neg_items))]
            ex_lst.append((user, pos_items, neg_items, past_items, rate_lst))
        return ex_lst
        
    def buildDatasetFromFile(self, dataset_fpath, max_batch_num_items, max_batch_num_pos, max_batch_num_negs, isshuffle=True):
        #read all examples
        all_ex_lst = self.make_all_lst(dataset_fpath, max_batch_num_pos, max_batch_num_negs)
        # call super class to read data
        self.buildDataset(all_ex_lst, max_batch_num_items, max_batch_num_pos, max_batch_num_negs, isshuffle)

if __name__ == '__main__':
    pass