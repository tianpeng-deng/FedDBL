# from msilib.schema import Feature
import os
import sys
import argparse
import pickle
from re import T
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
from copy import deepcopy
from functools import reduce

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.nn.functional as F

from tool.Logger import Logger
from tool.dataset import ImageDataset
from tool.DBL import DBL_net
from tool.resnet import resnet50
from tool.ctran import ctranspath
from tool.moco import builder_infence
from functools import partial
from tool import vits

from sklearn.metrics import accuracy_score,f1_score
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

seed = 2048
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
AGG_WEIGHT = {1: 0.28,
              2: 0.32,
              3: 0.33,
              4: 0.07}

# AGG_WEIGHT = {1: 0.368,
#               2: 0.435,
#               3: 0.197}          

def create_model(model_name, n_class, args):
    elif model_name == 'r50':
        ## Load ImageNet pretrained ResNet-50 backbone
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(512 * model.block.expansion, n_class)
    
    elif model_name == 'ctrans':
        ## Load CTransPath backbone
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(r'load/ctranspath.pth')
        model.load_state_dict(td['model'], strict=True)
    return model

def save_DBL(DBL, save_path, fname):
    with open(os.path.join(save_path, fname), 'wb') as f:
        pickle.dump(DBL, f)
def load_DBL(file_dir):
    with open(file_dir, 'rb') as f:
        DBL = pickle.load(f)
    return DBL

def train_DBL(model,trainset,args,DBL):
    model.eval()
    dataloader_train = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    steps = len(dataloader_train)
    dataiter_train = iter(dataloader_train)
    print('Training phase ---> number of training items is: ', len(trainset))
    work_space_in = np.zeros((len(trainset), args.n_feature))
    work_space_out = np.zeros((len(trainset), args.n_class))
    progress = 0
    for step in range(steps):
        img224, target_train = next(dataiter_train)
        len_batch = len(target_train)
        with torch.no_grad():
            img224 = Variable(img224.float().cuda())
            
        feature1, _ = model(img224)
        feature = feature1
        
        work_space_in[  progress:(progress+len_batch),  :] = feature.detach().cpu().numpy()
        work_space_out[ progress:(progress+len_batch),  :] = target_train.detach().cpu().numpy()
        progress = progress+len_batch
    DBL.train(work_space_in, work_space_out)
    return DBL

def valid_DBL(model, dataset, args, DBL):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    steps = len(dataloader)
    dataiter_valid = iter(dataloader)
    print('Validation phase ---> number of val items is: ', len(dataset))
    work_space_in = np.zeros((len(dataset), args.n_feature))
    work_space_out = np.zeros((len(dataset), args.n_class))
    progress = 0
    for step in tqdm(range(steps)):
        img224, target_valid = next(dataiter_valid)
        len_batch = len(target_valid)

        with torch.no_grad():
            img224 = Variable(img224.float().cuda())
            
        feature1, _ = model(img224)    
        feature = feature1
        
        work_space_in[progress:(progress+len_batch),:] = feature.cpu().detach().numpy()
        work_space_out[progress:(progress+len_batch),:] = target_valid.cpu().detach().numpy()
        progress = progress+len_batch
    ##  DBL prediction
    ##  In the prediction phase, DBL can also make predictions sample by sample.
    DBL_output =DBL.predict(work_space_in)
    DBL_pred = np.zeros(DBL_output.shape[0])
    DBL_lab = np.zeros(DBL_output.shape[0])
    for i in range(len(DBL_output)):
        DBL_pred[i] = np.argmax(DBL_output[i])
        DBL_lab[i] = np.argmax(work_space_out[i])
    DBL_acc = accuracy_score(DBL_lab,DBL_pred)
    DBL_f1 = f1_score(DBL_lab,DBL_pred,average='macro')

    return DBL_acc, DBL_f1
       
def main(args):
    ##  Training set and Validation set
    wholeVal = ImageDataset(data_path=args.wholeval, n_class=args.n_class)
    
    model_names = ['r50', 'ctrans']
    n_features = [3840, 768]
    for model_index in range(len(model_names)):
        args.model_name = model_names[model_index]
        
        print('<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('This model is', args.model_name)
    
        args.log_dir = args.save_dir + 'FedDBL_on_' + args.model_name + '.log'
        sys.stdout = Logger(os.path.join(args.log_dir), sys.stdout)
        
        args.n_feature = n_features[model_index]
        # ======================================================================
        # Load backbone
        model = create_model(model_name=args.model_name, n_class=args.n_class, args=args)
        model = model.cuda()

        DBL_dict = dict()
        for client in range(1, args.client_num + 1):
            trainset = ImageDataset(data_path = os.path.join(args.traindir, 'Client_' + str(client) + '/Train/'), n_class=args.n_class)
            validset = ImageDataset(data_path = os.path.join(args.validdir, 'Client_' + str(client) + '/Valid/'), n_class=args.n_class)
            print('Client {}, training samples {}'.format(client, len(trainset)))

            # Local DBL Training Phase
            reg = 0.001
            n_components = min(int(len(trainset)*0.9),  2000)
            DBL = DBL_net(isPCA=False, n_components = n_components, reg=reg)
            # ===================================================================
            # # Train & Save DBL
            DBL = train_DBL(model, trainset, args, DBL)
            save_DBL(DBL, args.save_dir, fname)
            # ===================================================================
            ## Load DBL
            # DBL = load_DBL(os.path.join(args.save_dir, fname))
            # ===================================================================
            
            DBL_dict[client] = DBL 
            train_acc, train_f1 = valid_DBL(model, trainset, args, DBL)
            valid_acc, valid_f1 = valid_DBL(model, validset, args, DBL)
            print('DBL: Client {} on Local trainset: ACC {}, F1 {}'.format(client, train_acc, train_f1))
            print('DBL: Client {} on Local validset: ACC {}, F1 {}'.format(client, valid_acc, valid_f1))
        
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Weighted Averaging for FedDBL')
        FedDBL_weight = np.zeros((args.n_feature, args.n_class))
        for client in range(1, args.client_num + 1):
            FedDBL_weight += DBL_dict[client].getWeight() * AGG_WEIGHT[client]
        
        reg = 0.001
        n_components = min(int(len(trainset)*0.9),  2000)
        FedDBL = DBL_net(isPCA=False, n_components = n_components, reg=reg)
        FedDBL.setWeight(FedDBL_weight)
        
        # Test Local Validation Accuracy and F1-score
        FedDBL_acc_train, FedDBL_f1_train = dict(), dict()
        FedDBL_acc_valid, FedDBL_f1_valid = dict(), dict()
        for client in range(1, args.client_num + 1):
            trainset = ImageDataset(data_path = os.path.join(args.traindir, 'Client_' + str(client) + '/Train/'), n_class=args.n_class)
            validset = ImageDataset(data_path = os.path.join(args.validdir, 'Client_' + str(client) + '/Valid/'), n_class=args.n_class)
            
            train_acc, train_f1 = valid_DBL(model, trainset, args, FedDBL)
            FedDBL_acc_train[client] = train_acc
            FedDBL_f1_train[client] = train_f1
            
            valid_acc, valid_f1 = valid_DBL(model, validset, args, FedDBL)
            FedDBL_acc_valid[client] = valid_acc
            FedDBL_f1_valid[client] = valid_f1
        
        print('FedDBL on all Trainsets')
        print('      ACC {}'.format(FedDBL_acc_train))
        print('      F1  {}'.format(FedDBL_f1_train))
        print('FedDBL on all Validsets')
        print('      ACC {}'.format(FedDBL_acc_valid))
        print('      F1  {}'.format(FedDBL_f1_valid))
        wholeVal_acc, wholeVal_f1 = valid_DBL(model, wholeVal, args, FedDBL)
        print('FedDBL on Whole Validset')
        print('      ACC {}'.format(wholeVal_acc))
        print('      F1  {}'.format(wholeVal_f1))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Federated Deep-Broad Learning')
    parser.add_argument('--device',     default='0',        type=str, help='index of GPU')
    parser.add_argument('--n_class',    default=9,          type=int, help='Number of categories')
    parser.add_argument('--n_workers',  default=0,          type=int, help='Number of workers')
    parser.add_argument('--client_num', default=4,          type=int,  help='Number of Training Clients')
    parser.add_argument('--lr',         default=0.001,      type=float,  help='Learning Rate')
    parser.add_argument('--batch_size', default=20,         type=int, help='Batch size of dataloaders')

    
    FOLDS = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
    PARTATIONS_FOLDS = ['001', '005', '010', '030', '050', '070', '100']
    SAVE_PATH       = 'YourPath/CRC/{}/FedDBL/{}/'
    DATASET_PATH    = 'YourPath/CRC/{}/{}/'
    
    for fold in FOLDS:
        for par_fold in PARTATIONS_FOLDS: 
            print("CURRENT PARTATION IS {}".format(par_fold))
            print("CURRENT FOLD IS {}".format(fold))
            args = parser.parse_args()
            args.save_dir = SAVE_PATH.format(par_fold, fold)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
                
            args.dataset_dir = DATASET_PATH.format(par_fold, fold)
            args.traindir   = args.dataset_dir
            
            valid_030 = DATASET_PATH.format('100', fold)
            args.validdir = valid_030
            args.wholeval   = os.path.join(valid_030, 'Centralized/Valid/')

            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            main(args)

    

