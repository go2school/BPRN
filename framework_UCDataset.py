# -*- coding:utf-8 -*-
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import random
import pandas as pd
import collections
import numpy as np
from framework_ndcg_UCDataset import UCSampler, UCDataset

##################################################################################################
# sampler dataset
##################################################################################################
class UCDataSampler(UCSampler):
    """
        negative sampling for pointwise and pairwise
    """
    def __init__(self, train_pos_fname, max_batch_num_pos, max_batch_num_items):
        # prepare training set (u, past_items, pos_items)
        # read data
        data = pd.read_csv(train_pos_fname)
        # split sequence A into A[:n] and A[n:]
        self.u_past_pos = self.make_user_past_positive_sequence(data, max_batch_num_pos)  
        # prepare user+item to path mapping      
        # set parameters
        self.max_batch_num_items = max_batch_num_items
        self.max_batch_num_pos = max_batch_num_pos        

    def make_user_past_positive_sequence(self, dt, num_pos):   
        #,eid,video,unit,section,chapter,course,type,wtime
        data_dp = dt.drop_duplicates(['eid', 'course'])
        #get (eid, past items). we can use this to make dataset
        eid_cid_seq = data_dp.groupby(by='eid')['course'].unique().to_frame().reset_index()
        ex_lst = []
        for tuple_ in eid_cid_seq.itertuples():
            u = tuple_.eid
            items = tuple_.course
            past_items = items[:-num_pos]
            pos_items = items[-num_pos:]
            #user,past_items,pos_items
            ex_lst.append([u, past_items, pos_items])
        return ex_lst

    def randomly_sample_negative_examples(self, u_past_pos, num_negative, all_items):
        ret_pos_and_neg_ex_lst = []
        for pos_ex in u_past_pos:
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
            # make 
            ret_pos_and_neg_ex_lst.append((user, pos_items, sampled_negs, train_items, rate_lst))
        return ret_pos_and_neg_ex_lst
    
    def randomly_sample_neg_ranking(self, all_items, num_positive, num_negative, isshuffle):
        user_pos_neg_past_rate_lst = self.randomly_sample_negative_examples(self.u_past_pos, num_negative, all_items)
        s = UCDataset()
        s.buildDataset(user_pos_neg_past_rate_lst, self.max_batch_num_items, num_positive, num_negative, isshuffle)
        return s    

class SeqUCDataset(UCDataset):
    def __init__(self):
        super()

    def make_all_lst(self, fname, neg_fname, num_pos, num_neg):
        # read eid to neg items
        eid2neg_cids = {}
        fd = open(neg_fname)
        for index, line in enumerate(fd):
            line = line.strip().split(',')
            if index >= 1:
                eid2neg_cids[int(line[0])] = [int(l) for l in line[1].split(' ')]
        fd.close()

        # read positive
        ex_lst = []
        dt = pd.read_csv(fname)
        # drop duplicate
        data_dp = dt.drop_duplicates(['eid', 'course'])
        #get (eid, past items). we can use this to make dataset
        eid_cid_seq = data_dp.groupby(by='eid')['course'].unique().to_frame().reset_index()

        for tuple_ in eid_cid_seq.itertuples():
            #user,items,neg_items
            user = tuple_.eid
            items = tuple_.course
            neg_items = eid2neg_cids[user][:num_neg]
            past_items = items[:-num_pos]
            pos_items = items[-num_pos:]
            rate_lst = [1 for i in range(num_pos)] + [0 for i in range(len(neg_items))]
            ex_lst.append((user, pos_items, neg_items, past_items, rate_lst))
        return ex_lst
    
    def buildDatasetFromFile(self, dataset_fpath, neg_item_dataset_fpath, max_batch_num_items, max_batch_num_pos, max_batch_num_negs, isshuffle=True):
        #read all examples
        all_ex_lst = self.make_all_lst(dataset_fpath, neg_item_dataset_fpath, max_batch_num_pos, max_batch_num_negs)
        # call super class to read data
        self.buildDataset(all_ex_lst, max_batch_num_items, max_batch_num_pos, max_batch_num_negs, isshuffle)

if __name__ == '__main__':
    pass