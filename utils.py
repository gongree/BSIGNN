import os
import torch
import numpy as np
import random
import csv
from sklearn import metrics

def gpu_setup(use_gpu=True, gpu_id=0):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

def set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

## Logger
class Logger:
    def __init__(self, k_fold=None, num_classes=None, type='train'):
        super().__init__()
        self.k_fold = k_fold
        self.num_classes = num_classes

        self.initialize(k=None)

        assert type in ['train' ,'test'], 'type은 train 또는 test 여야 합니다.'
        self.type = type

    def __call__(self, **kwargs):
        if len(kwargs)==0: 
            self.get()
        else: 
            self.add(**kwargs)
    
    def _initialize_metric_dict(self):
        return {"pred":[], "true":[], "prob":[]}

    def initialize(self, k=None):
        if self.k_fold is None:
            self.samples = self._initialize_metric_dict()
        else:
            if k is None:  # 전체 초기화
                self.samples = {}  # dict(dict)
                for i in range(1,self.k_fold+1):
                    self.samples[i] = self._initialize_metric_dict()
            else:
                self.samples[k] = self._initialize_metric_dict() 
    
    def add(self, k=None, **kwargs):
        if self.k_fold is None:
            for sample, value in kwargs.items():  # pred, true, prob
                self.samples[sample].append(value)
        else:
            assert k in list(range(1,self.k_fold+1))
            for sample, value in kwargs.items():
                self.samples[k][sample].append(value)

    def get(self, k=None, initialize=False):
        if self.k_fold is None:
            # value
            true = np.concatenate(self.sample['true'])
            pred = np.concatenate(self.sample['pred'])
            prob = np.concatenate(self.sample['prob'])
        else:
            if k is None:
                # dict -> key(k fold) : value(true, pred, prob)
                true, pred, prob = {}, {}, {}
                for k in range(1,self.k_fold+1):
                    # print(self.samples[k].keys())
                    true[k] = np.concatenate(self.samples[k]['true'])
                    pred[k] = np.concatenate(self.samples[k]['pred'])
                    prob[k] = np.concatenate(self.samples[k]['prob'])
            else:
                # k fold - value
                true = np.concatenate(self.samples[k]['true'])
                pred = np.concatenate(self.samples[k]['pred'])
                prob = np.concatenate(self.samples[k]['prob'])
            
        if initialize:
            self.initialize(k)  # 리셋! 초기화! 특정 k fold 딕셔너리 초기화!
        
        return dict(true=true, pred=pred, prob=prob)
    
    def evaluate(self, k=None, initialize=False, option='mean'):
        samples = self.get(k)  # k가 None이면 모든 fold값을 다 가져옴

        if not self.k_fold is None and k is None:
            if option=='mean': aggregate = np.mean
            elif option=='std': aggregate = np.std
            else: raise

            acc = aggregate([accuracy(samples['pred'][k], samples['true'][k]) for k in range(1,self.k_fold+1)])
            # precision = aggregate([metrics.precision_score(samples['true'][k], samples['pred'][k], average='weighted') for k in range(1,self.k_fold+1)])
            roc = aggregate([metrics.roc_auc_score(samples['true'][k], samples['prob'][k][:, 1]) for k in range(1,self.k_fold+1)]) if self.num_classes == 2 else np.mean([metrics.roc_auc_score(samples['true'][k], samples['prob'][k], average='macro', multi_class='ovr') for k in range(1,self.k_fold+1)])
            f1 = aggregate([metrics.f1_score(samples['true'][k], samples['pred'][k], average='weighted' if self.num_classes == 2 else 'weighted') for k in range(1,self.k_fold+1)])
           
            sen = aggregate([sensitivity(samples['pred'][k], samples['true'][k]) for k in range(1,self.k_fold+1)])
            spe = aggregate([specificity(samples['pred'][k], samples['true'][k]) for k in range(1,self.k_fold+1)])
        else:
            acc = accuracy(samples['pred'], samples['true'])
            # precision = metrics.precision_score(samples['true'], samples['pred'], average='weighted')
            roc = metrics.roc_auc_score(samples['true'], samples['prob'][:, 1]) if self.num_classes == 2 else metrics.roc_auc_score(samples['true'], samples['prob'], average='macro', multi_class='ovr')
            f1 = metrics.f1_score(samples['true'], samples['pred'], average='weighted' if self.num_classes == 2 else 'weighted')
            
            sen = sensitivity(samples['pred'], samples['true'])
            spe = specificity(samples['pred'], samples['true'])
        
        if initialize:
            self.initialize(k)

        if self.type == 'train':
            return dict([
                ('trn_acc', acc), 
                ('trn_roc_auc', roc), 
                ('trn_f1_score', f1), 
                ('trn_sensitivity', sen), 
                ('trn_specificity', spe)
            ])
        else:
            return dict(
                accuracy=acc, 
                roc_auc=roc, 
                f1_score=f1, 
                sensitivity=sen, 
                specificity=spe
            )
    
    def to_csv(self, targetdir, k=None, initialize=False):
        metric_dict = self.evaluate(k, initialize) 
        append = os.path.isfile(os.path.join(targetdir, f'metric.csv'))
        with open(os.path.join(targetdir, f'metric.csv'), 'a', newline='') as f:
            writer = csv.writer(f) 
            if not append:
                writer.writerow(['fold'] + [str(key) for key in metric_dict.keys()]) 
            writer.writerow([str(k)]+[str(value) for value in metric_dict.values()])
            if k is None:
                writer.writerow([str(k)]+list(self.evaluate(k, initialize, 'std').values()))

def accuracy(out, label):
    out = np.array(out)
    label = np.array(label)
    total = out.shape[0]
    correct = (out == label).sum().item() / total
    return correct
 
def sensitivity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label == 1.)
    sens = np.sum(out[mask]) / np.sum(mask)
    return sens
 
def specificity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label <= 1e-5)
    total = np.sum(mask)
    spec = (total - np.sum(out[mask])) / total
    return spec

