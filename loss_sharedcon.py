import torch
import torch.nn as nn
import pickle

### Credits https://github.com/HobbitLong/SupContrast
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature


    def forward(self, features, labels, mask=None):


        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        batch_size = features.shape[0] ## 2*N
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            contrast_count = 2
            anchor_count = contrast_count
            assert batch_size % 2 == 0
            mask = torch.eye(batch_size//2, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
        elif labels is not None:    # hsan: 우리가 쓰는거
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                print(labels.shape[0], "labels in loss file")
                print(batch_size, "batch_siae in loss file")
                raise ValueError('Num of labels does not match num of features')
            
            mask = torch.eq(labels, labels.T).float().to(device)

        else:
            raise NotImplementedError
        

        contrast_feature = features
        anchor_feature = contrast_feature #ywyw hsan-cl에서 느낌상 이거 contrast_feature랑 anchor_feature 따로 가면 될듯?


        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), 
            self.temperature)
        


        #ywyw 다시 살리기. 다시 살린 이유는 line 89-90 참고.
        # 기존 logits_mask 
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        
        ## hsan: change logits_mask
        temp_mask = torch.zeros_like(mask) #ywyw 바뀐 부분. 기존 logits_mask는 놔두고 새롭게 정의. (mask만드는 용도) 

        height, width = mask.shape
        half_height = height // 2
        half_width = width // 2

        temp_mask[:half_height, :half_width] = 0  
        temp_mask[half_height:, :half_width] = 1  
        temp_mask[:half_height, half_width:] = 1  
        temp_mask[half_height:, half_width:] = 0  


        ## it produces 1 for the non-matching places and 0 for matching places i.e its opposite of mask
        # mask = mask * logits_mask 
        mask = mask * temp_mask #ywyw 바뀐 부분. 기존과 기능은 동일.

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        logits = anchor_dot_contrast - logits_max.detach() 

        exp_logits = torch.exp(logits) * logits_mask    #  https://aclanthology.org/2021.emnlp-main.359.pdf 논문 (2) 식의 분모 exp(h_i h_k /t)에 해당하는 부분으로, 이 부분에서 곱해지는 logits_mask는 자기 자신만 제외하도록 하는 기능을 해야할 것 같습니다.
                                                        # temp_mask로 하면 자기 자신 뿐 아니라 2사분면 또는 4사분면에 해당하는 절반을 제외하게 됨.

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss


#######################################################################################
### Credits https://github.com/HobbitLong/SupContrast
class SupConLoss_for_double(nn.Module):

    def __init__(self, temperature=0.07):
        super(SupConLoss_for_double, self).__init__()
        self.temperature = temperature


    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] ## 3*N

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            contrast_count = 3
            anchor_count = contrast_count
            assert batch_size % 3 == 0
            mask = torch.eye(batch_size//3, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) 
        else:
            raise NotImplementedError

        contrast_feature = features
        anchor_feature = contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        ## it produces 1 for the non-matching places and 0 for matching places i.e its opposite of mask
        mask = mask * logits_mask 

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 

        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss