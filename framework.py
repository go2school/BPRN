# -*- coding:utf-8 -*-
import torch
# models
from framework_model_FISM import FISM
from framework_model_NAIS import NAIS
from framework_model_BPRN import BPRN
# data
from framework_UCDataset import UCDataSampler, SeqUCDataset
from util import read_str_to_id_map
from torch.utils.data import DataLoader
# sys
import argparse
import json
import time, sys
import numpy as np
import pickle
import collections
import pandas as pd

class Framework:
    def __init__(self, kwargs):
        self.device = kwargs['device']# cuda or cpu chosen for certain parameters
        self.model_name = kwargs['model_name']
        self.max_batch_num_items = kwargs['max_batch_num_items']
        self.all_items = set(kwargs['all_items'])# item+
        # for training
        self.numnegatives = kwargs['numnegatives']
        self.numpositives = kwargs['numpositives']
        # for testing
        self.num_test_positive = kwargs['num_test_positive']
        self.num_test_negative = kwargs['num_test_negative']
        # other
        self.samplingmethod = kwargs['samplingmethod']        

        # for computing loss        
        self.topk_lst = [1, 3, 5, 10]
        self.best_loss = -1
        self.best_hyper = None
        self.best_hyper_text = '' 

    def getTestBatch(self):        
        return 512

    def setHyperNames(self):        
        self.hyper_name_lst = ['lambda_']        

    def modelType(self, model_name):
        if self.model_name.endswith('bprn'):
            self.model_type = 'pairwise'              
        else:
            self.model_type = 'pointwise'

    def createModel(self, params):   
        if self.model_name == 'fism':
            model = FISM(params).to(self.device)
            model.init_weight()  
        elif self.model_name == 'fismbprn':  
            inner_model = FISM(params).to(self.device)                     
            model = BPRN(inner_model)
            model.init_weight()             
        elif self.model_name == 'nais':
            model = NAIS(params).to(self.device)
            model.init_weight()                         
        elif self.model_name == 'naisbprn':
            inner_model = NAIS(params).to(self.device)                     
            model = BPRN(inner_model)
            model.init_weight()                                 
        print(model.__class__, self.model_type, self.samplingmethod)
        return model

    def initTrainDataSampler(self, trainFpath):        
        self.sampler = UCDataSampler(trainFpath, self.numpositives, self.max_batch_num_items)         
            
    def randomlySampleTrainDataset(self, numnegatives, isshuffle):
        return self.sampler.randomly_sample_neg_ranking(self.all_items, self.numpositives, numnegatives, isshuffle)

    def readTestDataset(self, model_name, fpath, neg_fpath, isshuffle):
        s = SeqUCDataset()   
        s.buildDatasetFromFile(fpath, neg_fpath, self.max_batch_num_items, self.num_test_positive, self.num_test_negative, isshuffle)   
        return s

    def prepareBatch(self, tuple_name_lst, tuple_data_lst):
        batch_data = {}        
        for name, data in zip(tuple_name_lst, tuple_data_lst):
            batch_data[name] = data.to(self.device)
        return batch_data

    def resetExp(self):
        self.best_loss = -1
        self.best_hyper_text = ''
        self.best_hyper = None
        # for compute ndcg loss
        self.denominator_table = {}
        for topk in self.topk_lst:# + [self.num_test_positive + self.num_test_negative]:
            self.denominator_table[topk] = torch.from_numpy(np.log2( np.arange( 2, 2 + topk ))).float().to(self.device)
        # init max ndcg
        self.max_ndcg_score = {}
        for topk in self.topk_lst:# + [self.num_test_positive + self.num_test_negative]:
            actual_rels = torch.from_numpy(np.array([1 for i in range(self.num_test_positive)] + [0 for i in range(topk - self.num_test_positive)])).float().to(self.device)
            self.max_ndcg_score[topk] = (actual_rels[:topk] / self.denominator_table[topk]).sum().item() 

    def compute_ranking_loss(self, prediction_labels, true_labels):
        #compute batch ndcg, auc
        # out: (batch, n_test_docs)
        (_, sorted_idxs) = prediction_labels.sort(dim=1, descending=True)
        # compute NDCG HR
        ndcg_score_lst = {}
        hr_score_lst = {}
        for topk in self.topk_lst:
            # topk ranking
            topk_sorted_idxs = sorted_idxs[:, :topk]
            # find the relevance of item (i.e., 1 or 0)
            rel_topk_position = torch.gather(true_labels, 1, topk_sorted_idxs).float()
            # ndcg@topk
            # out: (batch)
            # NDCG@topk = \sum_i \frac{2^{rel_i}-1}{log_2(i+1)} / NDCG
            ndcg_score = (rel_topk_position / self.denominator_table[topk]).sum(1) / self.max_ndcg_score[topk]
            # hr@topk
            q = self.num_test_positive if topk > self.num_test_positive else topk
            hr_score = rel_topk_position.sum(1) / q
            # add lst            
            ndcg_score_lst[topk] = ndcg_score.tolist()
            hr_score_lst[topk] = hr_score.tolist()
        return ndcg_score_lst, hr_score_lst
        
    def computeLoss_(self, ndcg_score_map_lst, hr_loss_lst_map, hyper_parameter):
        hyper_text = ' '.join([hyname+':'+str(hyper_parameter[hyname]) for hyname in self.hyper_name_lst])
        test_ndcg_map_topk_val = {}        
        test_hr_map_topk_val = {}   
        str_ndcg_text = []
        str_hr_text = []
        for topk in self.topk_lst:
            test_ndcg_map_topk_val[topk] = np.mean(ndcg_score_map_lst[topk])
            test_hr_map_topk_val[topk] = np.mean(hr_loss_lst_map[topk])
            str_ndcg_text.append('test_NDCG@' + str(topk) + ' ' + ('%.4f' % test_ndcg_map_topk_val[topk]))        
            str_hr_text.append('test_HR@' + str(topk) + ' ' + ('%.4f' % test_hr_map_topk_val[topk]))
        #get best loss
        if test_ndcg_map_topk_val[10] > self.best_loss:
            self.best_loss = test_ndcg_map_topk_val[10]
            self.best_hyper = hyper_parameter
            self.best_hyper_text = hyper_text        
        return hyper_text, ' '.join(str_ndcg_text) + ' ' + ' '.join(str_hr_text) 
       
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.description='framework for course recommendation'
    parser.add_argument("-a","--alpha", default=0.5, help="the alpha parameter",type=float)
    parser.add_argument("-b","--batchsize", default=32, help="the batch size",type=int)    
    parser.add_argument("-d","--indir", default='', help="the dataset directory",type=str)
    parser.add_argument("-e","--beta", default=0.5, help="the nais beta parameter",type=float)
    parser.add_argument("-f","--numnegatives", default=4, help="number of negatives training examples",type=int)        
    parser.add_argument("-i","--iter", default=20, help="the number of iteration",type=int)
    parser.add_argument("-j","--hyperjsontext", default=None, help="the preset hyper paramter text body",type=str)
    parser.add_argument("-k","--samplingmethod", default="random", help="the refined sampling method",type=str)    
    parser.add_argument("-l","--learnrate", default=0.01, help="the learning rate",type=float)    
    parser.add_argument("-m","--model", default='fism', help="the model name",type=str)
    parser.add_argument("-n","--nfactors", default=16, help="the hidden dimension",type=int)    
    parser.add_argument("-p","--dropout", default=0.5, help="the dropout probablity",type=float)
    parser.add_argument("-q","--numpositives", default=1, help="number of positive training examples",type=int)    
    parser.add_argument("-t","--maxitems", default=50, help="the maximum number of items in a batch",type=int)        
    parser.add_argument("-x","--prediction", default="/home/xiao/tmp/pred", help="the path of prediction file",type=str) 
    parser.add_argument("-y","--numtestpositive", default=1, help="the positive items",type=int)  
    parser.add_argument("-z","--numtestnegative", default=99, help="the negative items",type=int)          
        
    args = parser.parse_args()
    print(str(args))    

    # for CPU/GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')

    # sys parameters
    num_iter_exp = args.iter    
    max_batch_num_items = args.maxitems         
    model_name = args.model
    # for hyper parameters
    batch_size = args.batchsize
    hyperjsontext = args.hyperjsontext       
    alpha_lst = [args.alpha]
    beta_lst_ = [args.beta]
    dropout_prob_lst = [args.dropout]    
    learning_rate_lst = [args.learnrate]
    n_factor_lst = [args.nfactors]                  
    # for IO path
    prediction_fpath = args.prediction
    # for train dataset
    indir = args.indir  
    samplingmethod = args.samplingmethod
    numnegatives = args.numnegatives
    numpositives = args.numpositives    
    # for testing dataset
    num_test_positive = args.numtestpositive
    num_test_negative = args.numtestnegative

    #for debugging
    if True:
        indir = './sample_dataset'
        model_names = ['fism', 'fismbprn', 'nais', 'naisbprn']        
        model_name = 'fism'
        n_factor_lst = [16]
        learning_rate_lst = [0.01]
        hyperjsontext = '{"lambda_": 1e-05}'        
        num_iter_exp = 10
        prediction_fpath = 'E:/tmp'
        samplingmethod = 'random'
        numpositives = 1
        numnegatives = 4
        num_test_positive = 1
        num_test_negative = 99           

    # dataset
    train_path = indir + '/train_0.csv'
    test_path = indir + '/test_0.csv'
    test_neg_path = indir + '/test_neg_0.csv'  
    
    # get max user ID and course ID
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    test_neg_data = pd.read_csv(test_neg_path)    
    all_users = set(train_data['eid']) | set(test_data['eid']) | set(test_neg_data['eid'])
    all_items = set(train_data['course']) | set(test_data['course'])
    for index, row in test_neg_data.iterrows():
        all_items |= set([int(k) for k in row['courses'].split(' ')])
    n_users = max(all_users) + 1
    n_items = max(all_items) + 1            
        
    print('#users', n_users, '#items', n_items)    

    kwargs = {}
    kwargs['device'] = device
    kwargs['model_name'] = model_name
    kwargs['max_batch_num_items'] = max_batch_num_items
    kwargs['all_items'] = all_items
    kwargs['numnegatives'] = numnegatives
    kwargs['numpositives'] = numpositives
    kwargs['num_test_positive'] = num_test_positive
    kwargs['num_test_negative'] = num_test_negative
    kwargs['samplingmethod'] = samplingmethod    

    framework = Framework(kwargs)
    framework.modelType(model_name)
    framework.setHyperNames()    
    framework.resetExp()
    #for negative sampler
    framework.initTrainDataSampler(train_path)
    # read test dataset
    testData = framework.readTestDataset(framework.model_name, test_path, test_neg_path, False)    
    print('#positive ex in training set', len(framework.sampler.u_past_pos), '#all test data', testData.len)
    
    basic_parameters = [[device], [n_users], [n_items], n_factor_lst, dropout_prob_lst, alpha_lst, beta_lst_, learning_rate_lst]
    basic_parameter_names = ['device', 'n_users', 'n_items', 'n_factors', 'dropout_prob', 'alpha', 'beta_', 'learning_rate']

    print('#users', n_users, '#items', n_items, 'load dataset_done')
    
    if hyperjsontext != None:
        param = dict(json.loads(hyperjsontext), **dict(zip(basic_parameter_names, [p[0] for p in basic_parameters])))
        print('preset hyper parameters', str(param))
    else:
        print('no hyper parameter. quit')
        sys.exit(0)    

    # create a model    
    param['numnegatives'] = numnegatives
    param['numpositives'] = numpositives
    param['max_batch_num_items'] = max_batch_num_items
    param['batch_size'] = batch_size
    param['maxidcg'] = framework.max_ndcg_score[numnegatives + numpositives]    
    
    model = framework.createModel(param)                
   
    #define loss and optimizer        
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'])
            
    #start learn
    start_tm = time.time()
    for iter in range(num_iter_exp):
        train_loss = 0
        train_reg = 0
        
        start_iter_train_tm = time.time()        
 
        # sampling negative examples and build training set  
        start_iter_evl_tm = time.time()                  
        trainData = framework.randomlySampleTrainDataset(numnegatives, True)                    
        # record time
        end_iter_evl_tm = time.time()
        if iter == 0:
            print('train set columns', trainData.len, trainData.explain_item())
            print('test set columns', testData.len, testData.explain_item())
        print('finish creating training dataset', len(trainData), '%.1f' % (end_iter_evl_tm - start_iter_evl_tm))          

        # start train                    
        model.train()
        batch_index = 0
    
        for batch_index, tuple_data in enumerate(DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)):              
            batch_data = framework.prepareBatch(trainData.explain_item(), tuple_data)
            # learn
            optimizer.zero_grad()
            
            train_prediction = model(batch_data)  
            reg = model.regularization()       
            if framework.model_type == 'pairwise':               
                loss = train_prediction + reg
            elif framework.model_type == 'pointwise':               
                loss = loss_fn(train_prediction, batch_data['rate_lst']) + reg
            
            train_loss += loss.item()            
            train_reg += reg.item()

            loss.backward()            
            optimizer.step()       
        
        train_loss /= (batch_index + 1)
        train_reg /= (batch_index + 1)
        # end train
        end_iter_train_tm = time.time()
        
        #do test   
        start_iter_test_tm = time.time()     
        model.eval()     
        #ndcg and hr loss        
        ndcg_loss_lst_map = {}
        hr_loss_lst_map = {}    
        for topk in framework.topk_lst:
            ndcg_loss_lst_map[topk] = []
            hr_loss_lst_map[topk] = []
        #bec loss
        prediction_lst = []
        user_lst = []
        for batch_index, tuple_data in enumerate(DataLoader(testData, batch_size=framework.getTestBatch(), num_workers=4, pin_memory=True)):
            # prepare a batch data            
            batch_data = framework.prepareBatch(testData.explain_item(), tuple_data)            
            # predict prediction
            # out: (batch, 100)
            test_prediction = model.predict(batch_data)
            if iter == num_iter_exp - 1:
                prediction_lst += test_prediction.view(-1).tolist()
            # true prediction
            # out: (batch, 100)
            true_prediction = batch_data['rate_lst']
            # compute ndcg, hr, auc for each user and return lists, bce is a score
            ndcg_score_lst, hr_score_lst = framework.compute_ranking_loss(test_prediction, true_prediction)            
            # append user and ndcg
            user_lst += batch_data['user'].tolist()
            # append ndcg and hr
            for topk in framework.topk_lst:
                ndcg_loss_lst_map[topk] += ndcg_score_lst[topk] 
                hr_loss_lst_map[topk] += hr_score_lst[topk]       
            del batch_data
            del test_prediction             
        end_iter_test_tm = time.time()   
        end_iter_tm = time.time()

        #evaluate average loss  
        hyper_output_str, test_loss_str = framework.computeLoss_(ndcg_loss_lst_map, hr_loss_lst_map, param)
        
        out_lst = [iter, hyper_output_str, 'train', '%.5f' %train_loss, 'train-reg', '%.5f' %train_reg, test_loss_str] + ['train_time', '%.1f' % (end_iter_train_tm - start_iter_train_tm)]  + ['test_time', '%.1f' % (end_iter_test_tm - start_iter_test_tm)]
        print(' '.join([str(s) for s in out_lst]))
        
        #write results
        if iter == num_iter_exp - 1:
            fw = open(prediction_fpath + '_' + str(iter), 'wb')
            pickle.dump(prediction_lst, fw)
            fw.close()

        # write user to ndcg
        fw = open(prediction_fpath + '_user_ndcg_10_' + str(iter)+'.csv', 'w')
        fw.write('user,ndcg@10\n')
        for i in range(len(user_lst)):
            fw.write(str(user_lst[i]) + ',' + str(ndcg_loss_lst_map[10][i]) + '\n')
        fw.close()
    
    end_tm = time.time()
    print('best_ndcg', framework.best_loss, 'using_time', '%.1f' % (end_tm - start_tm))    

    #store model
    if savemodelpath is not None:
        model.save_parameters(savemodelpath)
        print('save model', savemodelpath)
        