import argparse
import glob
import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import DatasetASD, DatasetMCI #, DatasetMDD
from utils import set_seed, Logger
from models import BSIGNN

import time
from setproctitle import *

setproctitle('fMRI_py')


### Steps ###
def train_step(model, optimizer, dataloader, logger, k, device, loss_weights, label_weight):

    model.train()
    epoch_loss = 0
    epoch_acc = 0
    num_data = 0
    true_pred = 0
    flag = True

    epoch_losses = [0,0,0,0]
    for i, data in enumerate(dataloader):
        x = data['bold'].to(device)
        y = data['label'].to(device)

        # print(x.shape)
        logits = model.forward(x)
        losses = model.calculate_loss(y, logits, *loss_weights, label_weight)

        loss = sum(loss_weights[z]*losses[z] for z in range(4))

        for z in range(4):
            epoch_losses[z] += losses[z].item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        prob = logits.detach() # .softmax(1)
        pred = logits.detach().argmax(dim=1)
        true_pred += (pred==y).float().sum().item()
        num_data += y.shape[0]

        logger.add(k=k, pred=pred.detach().cpu().numpy(), true=y.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
        if flag:
            tqdm.write(f"Pred {pred[:10]} | Label {y[:10]}")
            flag = False
            
    epoch_loss /= i+1
    epoch_acc = true_pred / num_data
    
    tqdm.write(f"Loss cls {epoch_losses[0]/(i+1):.2f} sparse {epoch_losses[1]/(i+1):.2f} rank {epoch_losses[2]/(i+1):.2f} mse {epoch_losses[3]/(i+1):.2f}")
    return epoch_loss, epoch_acc, optimizer, logger

def eval_step(model, dataloader, logger, k, device, loss_weights, label_weight):

    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    num_data = 0
    true_pred = 0
    edges = []
    labels = []

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            x = data['bold'].to(device)
            y = data['label'].to(device)

            logits = model.forward(x)
            losses = model.calculate_loss(y, logits, *loss_weights, label_weight)
            edges.append(model.brain_network)
            labels.append(y)

        epoch_loss += sum(loss_weights[z]*losses[z] for z in range(4)).item()
        pred = logits.detach().argmax(dim=1)
        prob = logits.detach()#.softmax(1)
        true_pred += (pred==y).float().sum().item()
        num_data += y.shape[0]
        tqdm.write(f"0 - pred {pred[pred==0].shape[0]} / real {y[y==0].shape[0]}")
        tqdm.write(f"1 - pred {pred[pred==1].shape[0]} / real {y[y==1].shape[0]}")

        logger.add(k=k, pred=pred.detach().cpu().numpy(), true=y.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())

    epoch_loss /= i+1
    epoch_acc = true_pred / num_data

    metric = logger.evaluate(k)
    metric['loss'] = epoch_loss

    return metric, logger, torch.concat(edges), torch.concat(labels)

# Main Trainer

def trainer(MODEL_NAME, DATASET_NAME, out_dir, params, profix, device):

    # Save directory
    root_log_dir = out_dir + 'logs/' + DATASET_NAME + '/log_' + profix
    root_ckpt_dir = out_dir + 'checkpoints/' + DATASET_NAME + '/checkpoint_' + profix
    root_final_dir = out_dir + 'final/' + DATASET_NAME + '/final_' + profix
    result_dir = out_dir + 'results/' + DATASET_NAME + '/result_' + profix

    # logger
    train_logger = Logger(k_fold=params['kfold'], num_classes=2, type='train')
    test_logger = Logger(k_fold=params['kfold'], num_classes=2, type='test')

    ## kfold
    if params['kfold'] == 1:
        print("K-Fold cross-validation is skipped because kfold is set to 1.")
    else:
        print(f"Performing {params['kfold']}-Fold Cross Validation\n\n")

    # Metrix
    acc, f1, sen, spe = [], [], [], []
    for k in range(1,6): # params['kfold']
        log_dir = os.path.join(root_log_dir, "RUN_" + "fold_"+str(k))

        run = wandb.init(project=f"{MODEL_NAME}", 
                        group=f"{DATASET_NAME}_{params['identify']}", 
                        name=f"{params['seed']}/fold_{k}", reinit=True)
        
        # initialize logger for k-th fold
        train_logger.initialize(k)
        test_logger.initialize(k)

        # load Dataset & Dataloader
        if DATASET_NAME=='ASD':
            train_dataset = DatasetASD(type='train',k=k)
            test_dataset = DatasetASD(type='test',k=k)
            label_weight = [0.443, 0.557]
        elif DATASET_NAME=='MCI':
            train_dataset = DatasetMCI(type='train',k=k)
            test_dataset = DatasetMCI(type='test',k=k)
            label_weight = [0.6219, 0.3781]
        # elif DATASET_NAME=='MDD':
        #     train_dataset = DatasetMDD(type='train',k=k)
        #     test_dataset = DatasetMDD(type='test',k=k)
        else:
            raise "Not ready for other datasets yet."

        train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True)

        # define Model
        num_nodes = train_dataset.num_nodes
        time_length = train_dataset.time_length
        model = BSIGNN(num_nodes=num_nodes, time_lag_length=8, input_dim=time_length, hidden_size=64, num_classes=2)
        model.to(device)

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(init_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay']) 

        loss_weights = [params["lambda_ce"],params["lambda_sparse"],params["lambda_rank"],params["lambda_mse"]]
        for epoch in tqdm(range(params['epochs']), desc=f"Training Fold {k}", ncols=80):
            epoch_train_loss, epoch_train_acc, optimizer, train_logger = train_step(model, optimizer, train_dataloader, train_logger, k=k, device=device, loss_weights=loss_weights, label_weight=label_weight)
            tqdm.write((f"Epoch{epoch:0{len(str(params['epochs']))}}/{params['epochs']} trn_loss {epoch_train_loss:.4f} trn_acc {epoch_train_acc:.4f}"))
            
            # wandb
            wandb.log({'train/_loss': epoch_train_loss, 'train/_acc': epoch_train_acc})
            
            # Saving checkpoint
            ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + "fold_"+str(k))
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

        # Save model and result per fold
        if not os.path.exists(root_final_dir):
                os.makedirs(root_final_dir)
        torch.save(model.state_dict(), f"{root_final_dir}/model_fold_{k}.pkl")
        
        test_metric, test_logger, learnable_edge, label = eval_step(model, test_dataloader, test_logger, k=k, device=device, loss_weights=loss_weights, label_weight=label_weight)
        
        print(f"\n\nFinal Result\nacc {test_metric['accuracy']:0.4f} sen {test_metric['sensitivity']:0.4f} "
              f"spe {test_metric['specificity']:0.4f} f1 {test_metric['f1_score']:0.4f} loss {test_metric['loss']:0.4f}")
        
        acc.append(test_metric['accuracy'])
        f1.append(test_metric['f1_score'])
        sen.append(test_metric['sensitivity'])
        spe.append(test_metric['specificity'])

        wandb.log(test_metric)
        run.finish()

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        torch.save(learnable_edge, '{}.pkl'.format(result_dir + "/learnable_edge_" + str(k)))
        torch.save(label, '{}.pkl'.format(result_dir + "/label_" + str(k)))

    
    # Final Result
    final_metrics = test_logger.evaluate() 
    print(f"\n\n[id-{params['identify']}] seed {params['seed']} : 5-fold Result \n")
    print("acc {:0.2f}±{:0.2f} f1 {:0.2f}±{:0.2f} sen {:0.2f}±{:0.2f} spe {:0.2f}±{:0.2f}"\
          .format(np.mean(np.array(acc)*100),np.std(np.array(acc)*100),
                  np.mean(np.array(f1)*100),np.std(np.array(f1)*100),
                  np.mean(np.array(sen)*100),np.std(np.array(sen)*100),
                  np.mean(np.array(spe)*100),np.std(np.array(spe))*100))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    test_logger.to_csv(result_dir)
    for i in range(1,6):
        test_logger.to_csv(result_dir, k=i)


if __name__=="__main__":

    # ArgumentParser
    parser = argparse.ArgumentParser(description="This script processes train, valid and test for dynamic FC.")
    parser.add_argument('-c', '--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('-s', '--seed', help='Please give a value for seed')
    parser.add_argument('-g', '--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('-d', '--dataset', help="Please give a value for dataset name")
    parser.add_argument('-m', '--model', help="Please give a value for model name")
    args = parser.parse_args()

    ## Configure
    if args.config is None:
        args.config = "configs/basic_config.json"
    with open(args.config) as f:
        config = json.load(f)

    ## setting the device and seed number 
    if args.gpu_id is None:
        config['gpu']=False
        device = torch.device('cpu')
        print("\n\nUsing device: cpu")
    else:
        assert args.gpu_id in ['6', '7'], "사용할 수 있는 gpu id가 아닙니다."
        config['gpu']['id'] = args.gpu_id

        # gpu_id = '6'  # or 7
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_gpu = torch.cuda.current_device()

        print(f"\n\nUsing device: {device}, GPU index: {args.gpu_id}")
        print(f"GPU name: {torch.cuda.get_device_name(current_gpu)}")

    config['params']['identify'] = datetime.now().strftime('%y%m%d%H%M%S')
    print(f"IDENTIFY : {config['params']['identify']}")

    # seed
    if args.seed is not None:
        config['params']['seed'] = int(args.seed)
    print("Seed : ",config['params']['seed'])
    set_seed(config['params']['seed'])

    ## Set the model and dataset
    if args.model is not None:
        MODEL_NAME = args.model
        config['model'] = MODEL_NAME
    else:
        MODEL_NAME = config['model']
    print("MODEL_NAME : ",MODEL_NAME)

    if args.dataset is not None:
        DATASET_NAME = args.dataset
        config['dataset'] = DATASET_NAME
    else:
        DATASET_NAME = config['dataset']
    print("DATASET_NAME : ",DATASET_NAME)
    

    ## Saving
    out_dir = config['out_dir']
    
    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    if not os.path.exists(out_dir + 'logs'):
        os.makedirs(out_dir + 'logs')

    if not os.path.exists(out_dir + 'checkpoints'):
        os.makedirs(out_dir + 'checkpoints')

    profix = MODEL_NAME + "_seed" + str(config['params']['seed']) + "_" + config['params']['identify']

    write_config_file = out_dir + 'configs/' + DATASET_NAME + '/' + 'config_' +  profix
    
    if not os.path.exists(write_config_file):
        os.makedirs(write_config_file)
    with open(os.path.join(write_config_file,"config.json"), "w") as f:
        json.dump(config, f)
    
    trainer(MODEL_NAME, DATASET_NAME, out_dir, config['params'], profix, device)

    