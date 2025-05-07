import os
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
from models.lsan import LSAN  # Ensure correct import path
from metric import cal_measures
from dataloaders.dataloader_seqrs import DataLoader_seq

torch.set_num_threads(1)

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.data_loader = DataLoader_seq(self.opt)
        self.trn_loader, self.vld_loader, self.tst_loader = self.data_loader.get_loaders()

        opt.numuser = self.trn_loader.dataset.numuser
        opt.numitem = self.trn_loader.dataset.numitem
        self.model = LSAN(self.opt).cuda()

        self._print_args()

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        best_score = -1
        for epoch in range(self.opt.num_epoch):
            self.model.train()
            epoch_loss = 0
            for batch_data in self.trn_loader:
                batch_data = [bd.cuda() for bd in batch_data]
                optimizer.zero_grad()
                loss = self.model.compute_loss(batch_data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.opt.num_epoch}, Loss: {epoch_loss:.4f}")

            # Validation
            with torch.no_grad():
                topHits, topNdcgs = cal_measures(self.vld_loader, self.model, self.opt, 'vld')
                score = (topHits[10] + topNdcgs[10]) / 2
                if score > best_score:
                    best_score = score
                    print(f"New best score: {best_score:.4f}")

    def _print_args(self):
        print("> Training arguments:")
        for arg in vars(self.opt):
            print(f">>> {arg}: {getattr(self.opt, arg)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lsan', type=str)
    parser.add_argument('--dataset', default='tools', type=str)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu)

    opt.dataset_path = f'./data/{opt.dataset}/rec'
    ins = Instructor(opt)
    ins.train()
