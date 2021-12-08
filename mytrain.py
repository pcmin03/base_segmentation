import numpy as np
import skimage 
import os ,tqdm , random
from glob import glob


import torch
from utils.logger import Logger

from datacode.mydataset import make_dataset,BuildDataset
from models import init_model
from losses import init_loss
from trainer import Trainer

from torch.utils.data import Dataset, DataLoader
import wandb
import argparse 
def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
    

def main(): 

    
    parser = argparse.ArgumentParser(description='Process some integers')
    parser.add_argument('--gpu', default='0',help='comma separated list of GPU(s) to use.',type=str)
    parser.add_argument('--weight_decay',default=1e-5,help='set weight_decay',type=float)

    # optimizer scheduler
    parser.add_argument('--optim',default='Adam',help='select optimizers method',type=str)
    parser.add_argument('--scheduler',default='Cosine',help='select schduler method',type=str)
    parser.add_argument('--start_lr',default=3e-3, help='set of learning rate', type=float)
    parser.add_argument('--end_lr',default=3e-5,help='set fo end learning rate',type=float)
    
    # dataset parameter
    parser.add_argument('--base_path',default='../',help='set base path',type=str)
    parser.add_argument('--Kfold',default=0,help='select kth fold ',type=int)
    
    # model parameter
    parser.add_argument('--out_class',default=1,help='set of output class',type=int)
    parser.add_argument('--encoder',default='resnet34',help='select model',type=str)
    parser.add_argument('--decoder',default='unet',help='select model',type=str)
    parser.add_argument('--pretrain_name',default='imagenet',type=str)
    parser.add_argument('--image_size',default=512,type=int)
    
    # loss paramter 
    parser.add_argument('--lossname',default='jaccard',help='select model',type=str)
    
    # hyperparamter
    parser.add_argument('--batch_size', default=10,help='set a batch size',type=int)
    parser.add_argument('--epochs',default=201,help='epochs',type=int)
    
    # start evaluation time
    parser.add_argument('--save_step',default=10,help='every epoch time vlidation ',type=int)
    
    # training setting
    parser.add_argument('--exp_name',default='exp',type=str)

    args = parser.parse_args()
    
    #set devices
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    print(torch.cuda.is_available(),'torch.cuda.is_available()')
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'else')
    
    print(torch.cuda.get_device_name(0))
    
    # seed seeting
    set_seed(seed=42)

    #inport network
    model = init_model(args) # custom model 
    
    #select loss
    loss = init_loss(args) # custom model 
 
    # make dataset
    train_df,valid_df = make_dataset(args)

    train_dataset = BuildDataset(train_df, transforms='train')
    valid_dataset = BuildDataset(valid_df, transforms='valid')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, 
                              num_workers=4, shuffle=False, pin_memory=True)

    #set log 
    #logger = Logger(main_path,valid_path+lossname,delete=args.deleteall)
    logger =  wandb.init(project='sartorius-public', 
                config={k:v for k, v in dict(vars(args)).items() if '__' not in k},
                anonymous='must',
                name=f"fold-{args.Kfold}|dim-{args.image_size}|model-{args.decoder}",
                group=args.exp_name,
            )

    #import trainer
    Learner = Trainer(model,train_loader,valid_loader,loss,logger,args,device)

    

    # if args.use_train == True: 
    Learner.run()
    
if __name__ == '__main__': 
    main()
