import numpy as np
import os ,tqdm , random
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch import Tensor
#custom set#

from utils.matrix import Evaluator, AverageMeter
import torch.autograd as autograd

from utils.pytorchtools import EarlyStopping
import skimage
from torch.cuda import amp
import segmentation_models_pytorch as smp

import time
import copy
from collections import defaultdict

class Trainer:
    total_train_iter = 0
    total_valid_iter = 0
    
    best_epoch = 0 
    best_axon_recall = 0
    
    def __init__(self,model,train_loader,valid_loader,loss,logger,args,device):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.logger = logger
        self.loss = loss.to(device)
        
        self.args = args
        self.epochs = args.epochs
        self.device = device
        
        # LR 
        self.optimizer = optim.Adam(self.model.parameters(),lr=args.start_lr,weight_decay=args.weight_decay)    
        # evaluator
        self.evaluator = Evaluator(args.out_class)
        # scheuler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,2,T_mult=2,eta_min=args.end_lr)

        self.t_loss = AverageMeter(args.out_class)
        self.recon_loss = AverageMeter(args.out_class)

        # self.early_stopping = EarlyStopping(patience = 10, verbose = True,save_path=self.logger.log_dir)

        self.JaccardLoss = smp.losses.JaccardLoss(mode='binary')
        self.Jaccard     = smp.losses.JaccardLoss(mode='binary', from_logits=False)
        self.Dice        = smp.losses.DiceLoss(mode='binary', from_logits=False)
        self.BCELoss     = smp.losses.SoftBCEWithLogitsLoss()


    def train_one_epoch(self,epoch):
        
        # reset loss value to zero. 
        self.model.train()
        # self.loggerning_loss = 0
        scaler = amp.GradScaler()
        dataset_size = 0 
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Train ')
        for step,(images,masks) in pbar:
            self.t_loss.reset_dict()

            images = images.to(self.device, dtype=torch.float)
            masks  = masks.to(self.device, dtype=torch.float)
            # self.mask_ = batch[2].to(self.device).unsqueeze(0)
            # self.bodygt = batch[3].to(self.device).unsqueeze(0)
            with amp.autocast(enabled=True):
                y_pred = self.model(images)
                loss   = self.loss(y_pred, masks)
                scaler.scale(loss).backward()
                
                scaler.step(self.optimizer)
                scaler.update()
            
                if self.scheduler is not None:
                    self.scheduler.step()
            # backpropagation
            
            # self.loggerning_loss += (loss.item() * self.args.batch_size)
            # dataset_size += self.args.batch_size

            epoch_loss = loss.item() 


            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                            lr=self.optimizer.param_groups[0]['lr'],
                            gpu_memory=f'{mem:0.2f} GB')

        return epoch_loss

    @torch.no_grad()
    def valid_one_epoch(self,epoch):

        self.model.eval()
        
        dataset_size = 0
        self.loggerning_loss = 0.0
        
        targets = []
        predicts   = []
        
        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader ), desc='Valid ')
        for step, (images, masks) in pbar:        
            images  = images.to(self.device, dtype=torch.float)
            masks   = masks.to(self.device, dtype=torch.float)
            
            batch_size = images.size(0)
            
            y_pred  = self.model(images)
            loss    = self.loss(y_pred, masks)
            
            self.loggerning_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            
            epoch_loss = self.loggerning_loss / dataset_size
            
            targets.append(nn.Sigmoid()(y_pred))
            predicts.append(masks)
            
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            
            pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                            lr=self.optimizer.param_groups[0]['lr'],
                            gpu_memory=f'{mem:0.2f} GB')
        
        targets = torch.cat(targets,dim=0).to(torch.float32)
        predicts   = (torch.cat(predicts, dim=0)>0.5).to(torch.float32)
        val_dice    = 1. - self.Dice(targets, predicts).cpu().detach().numpy()
        val_jaccard = 1. - self.Jaccard(targets, predicts).cpu().detach().numpy()
        val_scores  = [val_dice, val_jaccard]
        
        return epoch_loss, val_scores
  
    def run(self):
        # To automatically log gradients
        self.logger.watch(self.model, log_freq=100)
        
        if torch.cuda.is_available():
            print("cuda: {}\n".format(torch.cuda.get_device_name()))
        
        start = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_dice  = -np.inf
        best_epoch = -1
        history = defaultdict(list)
        
        for epoch in range(1, self.args.epochs + 1): 
            
            print(f'Epoch {epoch}/{self.args.epochs}', end='')
            train_loss = self.train_one_epoch(epoch)
            
            val_loss, val_scores = self.valid_one_epoch(epoch)
            val_dice, val_jaccard = val_scores
        
            history['Train Loss'].append(train_loss)
            history['Valid Loss'].append(val_loss)
            history['Valid Dice'].append(val_dice)
            history['Valid Jaccard'].append(val_jaccard)
            
            # Log the metrics
            self.logger.log({"Train Loss": train_loss, 
                    "Valid Loss": val_loss,
                    "Valid Dice": val_dice,
                    "Valid Jaccard": val_jaccard,
                    "LR":self.scheduler.get_last_lr()[0]})
            
            print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
            
            # deep copy the model
            if val_dice >= best_dice:
                print(f"Valid Dice Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
                best_dice    = val_dice
                best_jaccard = val_jaccard
                best_epoch   = epoch
                self.logger.summary["Best Dice"]    = best_dice
                self.logger.summary["Best Jaccard"] = best_jaccard
                self.logger.summary["Best Epoch"]   = best_epoch
                best_model_wts = copy.deepcopy(self.model.state_dict())
                PATH = f"best_epoch-{self.args.Kfold:02d}.bin"
                torch.save(self.model.state_dict(), PATH)
                # Save a model file from the current directory
                self.logger.save(PATH)
                print(f"Model Saved")
                
            last_model_wts = copy.deepcopy(self.model.state_dict())
            PATH = f"last_epoch-{self.args.Kfold:02d}.bin"
            torch.save(self.model.state_dict(), PATH)
                
            print(); print()
        
        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        print("Best Score: {:.4f}".format(best_dice))
        
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        