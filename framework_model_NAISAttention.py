# -*- coding:utf-8 -*-
import torch
import numpy as np

class NAISAttention(torch.nn.Module):    
    def __init__(self, dimensions, device, beta, lambda_, activation):
        super(NAISAttention, self).__init__()

        self.device = device
        self.beta = beta
        self.h = torch.nn.Parameter(torch.FloatTensor(dimensions, 1)).to(self.device) #h in attention
        self.linear_layer = torch.nn.Linear(dimensions, dimensions, bias=True)# W and b in MLP layer
        self.softmax = torch.nn.Softmax(dim=-1)
        self.activation = activation
        self.lambda_ = lambda_

    def init_weight(self):
        self.linear_layer.weight.data.normal_(0, self.lambda_)
        self.h.data.normal_(0, self.lambda_)

    def save_parameters(self, path):
        pass

    def load_parameters(self, fpath):
        pass

    def forward(self, query, context, context_mask):             
        batch_size, item_len, dimensions = context.size()        

        # mask context
        # In (batch size, item_length)
        # Out (batch size, item_length, 1)
        context_masked = context * context_mask.unsqueeze(2)

        # element-wise matrix product
        # In (batch_size, 1, dimensions) * (batch_size, item_len, dimensions) ->
        # Out (batch_size, item_len, dimensions)
        pq = query.unsqueeze(1) * context_masked #(batch_size, item_len, dimensions)
        pq_ = pq.view(-1, dimensions)#(batch * item_len, dimensions)

        # a linear layer
        # In (batch_size * item_len, dimensions)
        # Out (batch_size * item_len, dimensions)
        linear_output = self.linear_layer(pq_)
        if self.activation == 0:
            linear_output_ = torch.relu(linear_output)
        elif self.activation == 1:
            linear_output_ = torch.sigmoid(linear_output)
        elif self.activation == 2:
            linear_output_ = torch.tanh(linear_output)

        # attention score
        # In (batch_size * item_len, dimensions), (dimensions, 1)
        # Out (batch_size * item_len, 1)
        # reshape tensor
        # Out (batch_size, item_len)
        A_1 = torch.mm(linear_output_, self.h).view(batch_size, item_len)

        # use a mask to filter data
        # keep mask(1) and clear mask(0)
        # In (batch_size, item_len) and (batch_size, item_len)
        # Out (batch_size, item_len)        
        A = A_1 * context_mask

        # through softmax for normalization
        # In (batch_size, item_len)
        # Out (batch_size, item_len)
        if self.beta == 1:
            attention_weight = self.softmax(A)#(5, 7)
            attention_weight_ = attention_weight.unsqueeze(2)
        else:
            #compute softmax on non-zero rows
            # do mask
            A_without_zero = A.sum(1) != 0
            if A_without_zero.sum() != A.shape[0]:#exist zero row                
                A_rest = A[A_without_zero.to(torch.device('cpu'))]            
                context_mask_rest = context_mask[A_without_zero.to(torch.device('cpu'))]
                # compute upper part of \frac{exp(f(i,j))}{\sum_k exp(f(i,j))}
                A_ = torch.exp(A_rest) * context_mask_rest
                # compute lower part 
                smoothing_softmax_denominator = A_.sum(1).pow(-self.beta) 
                smoothing_softmax_denominator_ = smoothing_softmax_denominator.unsqueeze(1)
                # compute \frac{exp(f(i,j))}{\sum_k exp(f(i,j))}
                attention_weight = A_ * smoothing_softmax_denominator_
                # restore results
                attention_weight_ = torch.zeros(A.shape).to(self.device)
                attention_weight_[A_without_zero.to(torch.device('cpu'))] = attention_weight
                # In (batch_size, item_len)
                # Out (batch_size, item_len, 1)
                attention_weight_ = attention_weight_.unsqueeze(2)
            else:
                # a_ij = \frac{exp(f(p_i, q_j))}{(\sum_j exp(f(p_i, q_j)))^beta}
                # this step multiply mask is important in order to avoid unnecessary sum
                A_ = torch.exp(A) * context_mask                
                # get sum on denominator
                smoothing_softmax_denominator = A_.sum(1).pow(-self.beta)                            
                # In (batch)
                # Out (batch, 1)
                smoothing_softmax_denominator_ = smoothing_softmax_denominator.unsqueeze(1)
                # if A[i,:] is all zero, the wegiht will be inf large, so filter them again.
                attention_weight = A_ * smoothing_softmax_denominator_
                # In (batch_size, item_len)
                # Out (batch_size, item_len, 1)
                attention_weight_ = attention_weight.unsqueeze(2)#(5, 7, 1)

        # set up final attended score by element-wise matrix production
        # In (batch_size, item_len, 1) * (batch_size, item_len, dimensions)
        # Out (batch_size, item_len, dimensions)
        ret = attention_weight_ * context_masked#(5, 7, 10)

        # sum all items
        # In (batch_size, item_len, dimensions)
        # Out (batch_size, dimensions)
        output = ret.sum(1)#(5, 10), a final result

        return output, attention_weight

    def regularization(self, gama_, eta_):
        l2_h_regularization = gama_  * torch.norm(self.h.data, 2) ** 2
        l2_W_regularization = eta_ * torch.norm(self.linear_layer.weight, 2) ** 2
        l2_bias_regularization = eta_ * torch.norm(self.linear_layer.bias.data, 2) ** 2
        return l2_h_regularization + l2_W_regularization + l2_bias_regularization

if __name__ == '__main__':
    pass