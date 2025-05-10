import os
import time
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import numpy as np
import argparse
import random
import pickle
from models import LSAN
from metric import cal_measures, get_each_score, get_logit, get_pis
from dataloaders.dataloader_L_sequential import DataLoader as DataLoader_LSQ

torch.set_num_threads(1)

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.data_loader = DataLoader_LSQ(self.opt)
        self.trn_loader, self.vld_loader, self.tst_loader = self.data_loader.get_loaders()
        
        opt.numuser = self.trn_loader.dataset.numuser
        opt.numitem = self.trn_loader.dataset.numitem
        self.model = LSAN(self.opt).cuda()
        
        self._print_args()
        
    def train(self):        
        newtime = round(time.time())        
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.learning_rate) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
     
        best_score = -1 
        best_topHits, best_topNdcgs, best_topAccs = None, None, None
        batch_loss = 0
        c = 0 # to check early stopping
        
        for epoch in range(self.opt.num_epoch):
            st = time.time()
    
            for i, batch_data in enumerate(self.trn_loader):
                batch_data = [bd.cuda() for bd in batch_data]                                 
                optimizer.zero_grad() 
                
                loss = self.model.compute_loss(batch_data)                
                    
                loss.backward()
                
                optimizer.step()
    
                batch_loss += loss.data.item()

            elapsed = time.time() - st
            evalt = time.time()
            
            with torch.no_grad():
                topHits, topNdcgs, topAccs = cal_measures(self.vld_loader, self.model, opt, 'vld')                
                validation_score = (topHits[10] + topNdcgs[10]) / 2
                
                scheduler.step(validation_score)
                
                if validation_score > best_score:
                    best_score = validation_score
                    
                    best_topHits = topHits
                    best_topNdcgs = topNdcgs
                    best_topAccs = topAccs
                    
                    c = 0
                    
                    test_topHits, test_topNdcgs, test_topAccs = cal_measures(
                                    self.tst_loader, self.model, opt, 'tst')
                    
                    if opt.save == True:          
                        torch.save(self.model.ebd_user.weight, 
                                   opt.save_path+'/useremb_{}_{}.pth'.format(opt.model_name, opt.dataset))
                        torch.save(self.model.ebd_item.weight, 
                                   opt.save_path+'/itememb_{}_{}.pth'.format(opt.model_name, opt.dataset))

                        logit = get_logit(self.tst_loader, self.model, opt, 'tst')
                        np.save(opt.save_path+'/tstlogit_{}_{}.npy'.format(opt.model_name, opt.dataset), logit)

                    
                evalt = time.time() - evalt 
            
            print(('(%.1fs, %.1fs)\tEpoch [%d/%d], TRN_ERR : %.4f, v_score : %5.4f, tHR@10 : %5.4f, tAcc@10 : %5.4f'% 
                (elapsed, evalt, epoch, self.opt.num_epoch, batch_loss/len(self.trn_loader), validation_score, test_topHits[10], test_topAccs[10])))

            batch_loss = 0

            c += 1
            
            if c > 5: break # Early-stopping
        
        print(('\nValid score@10 : %5.4f, HR@10 : %5.4f, NDCG@10 : %5.4f, ACC@10 : %5.4f\n'% 
            (((best_topHits[10] + best_topNdcgs[10])/2), best_topHits[10],  best_topNdcgs[10], best_topAccs[10])))
        
        return test_topHits,  test_topNdcgs, test_topAccs, best_score, best_topHits[10], best_topNdcgs[10], best_topAccs[10]
            
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('\nn_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
        print('')

    def run(self, repeats):
        results = []
        rndseed = [312596, 970456, 303598, 970542, 513897] # random seeds
        for i in range(repeats):
            print('\n💫 run: {}/{}'.format(i+1, repeats))
            
            random.seed(rndseed[i]); np.random.seed(rndseed[i]); torch.manual_seed(rndseed[i])            

            self.model = LSAN(self.opt).cuda()
            
            results.append(ins.train())
        
        results = np.array(results)
        
        best_vld_scores = results[:,3].mean()
        best_vld_HR = results[:,4].mean()
        best_vld_nDCG = results[:,5].mean()
        best_vld_ACC = results[:,6].mean()
        print('\nBest VLD scores (mean): {:.4}\tHR@10:\t{:.4}\tnDCG@10:\t{:.4}\tACC@10:\t{:.4}\n'.format(best_vld_scores, best_vld_HR, best_vld_nDCG, best_vld_ACC))
        
        hrs_mean = np.array([list(i.values()) for i in results[:,0]]).mean(0)
        ndcg_mean = np.array([list(i.values()) for i in results[:,1]]).mean(0)
        acc_mean = np.array([list(i.values()) for i in results[:,2]]).mean(0)
        
        hrs_std = np.array([list(i.values()) for i in results[:,0]]).std(0)
        ndcg_std = np.array([list(i.values()) for i in results[:,1]]).std(0)
        acc_std = np.array([list(i.values()) for i in results[:,2]]).std(0)
        
        print('*TST STD\tTop2\tTop5\t\tTop10\t\tTop20\t')
        print('*HR means: {}'.format(', '.join(hrs_std.astype(str))))
        print('*NDCG means: {}'.format(', '.join(ndcg_std.astype(str))))
        print('*ACC means: {}\n'.format(', '.join(acc_std.astype(str))))
        
    
        print('*TST Performance\tTop2\tTop5\t\tTop10\t\tTop20\t')
        print('*HR means: {}'.format(', '.join(hrs_mean.astype(str))))
        print('*NDCG means: {}'.format(', '.join(ndcg_mean.astype(str))))
        print('*ACC means: {}'.format(', '.join(acc_mean.astype(str))))
        
    def _reset_params(self):
        self.model = LSAN(self.opt).cuda()
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lsan', type=str)
    parser.add_argument('--dataset', default='tools', type=str)    
    parser.add_argument('--num_run', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)    
    parser.add_argument('--batch_size', default=256, type=int)    
    parser.add_argument('--save', default=False, type=str2bool)
    parser.add_argument('--num_worker', default=8, type=int)    

    # HPs for general RS
    parser.add_argument('--margin', default=0.6, type=float)    
    parser.add_argument('--K', default=128, type=int)      
    parser.add_argument('--numneg', default=10, type=int)
    parser.add_argument('--lamb', default=0.5, type=float) # Equation 7 and 8
    parser.add_argument('--mu', default=0.3, type=float)  # Equation 7 and 8

    parser.add_argument('--binsize', default=8, type=int) # $w$ in the paper (section 3.2.2)
    parser.add_argument('--period', default=64, type=int) # It controls $T$ in the paper (section 3.2.2)
    parser.add_argument('--tau', default=0, type=float) # $\tau$ in the paper (section 3.3)
    parser.add_argument('--neg_weight', default=1.0, type=float)
    parser.add_argument('--bin_ratio', default=0.5, type=float)    

    parser.add_argument('--warmup_epochs', default=20, type=int)
    parser.add_argument('--aggtype', default='max', type=str, help='sum, mean, max')
    parser.add_argument('--maxhist', default=100, type=int) # The maximum # of consumed items per user
    parser.add_argument('--dropout', default=0.3, type=float) 
    parser.add_argument('--num_layer', default=2, type=int) 
    parser.add_argument('--num_next', default=1, type=int) 
    parser.add_argument('--kernel_size', default=5, type=int)  

    opt = parser.parse_args()   

    torch.cuda.set_device(opt.gpu)
    
    opt.model_class = LSAN
    opt.dataset_path = './data/{}/rec'.format(opt.dataset)
    opt.save_path = './trained_models/'

    ins = Instructor(opt)
    
    ins.run(opt.num_run)
