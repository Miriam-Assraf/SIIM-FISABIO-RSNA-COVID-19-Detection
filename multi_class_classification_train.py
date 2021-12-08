#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from glob import glob
import timm
import torchmetrics 
import matplotlib.pyplot as plt
# --- images --- 
import cv2
import albumentations as A
# --- time ---
from datetime import datetime
import time
# --- data ---
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from sklearn.model_selection import StratifiedKFold
# --- wandb ---
import wandb
from kaggle_secrets import UserSecretsClient
# --- dicom ---
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
# --- warnings ---
import warnings
warnings.filterwarnings('ignore')


# <a id="section-two"></a>
# ## **Basic configuration**

# In[7]:


# --- configs ---
NEGATIVE = 'negative'
TYPICAL = 'typical'
INDERTEMINATE = 'indeterminate'
ATYPICAL = 'atypical'

class Configs:
    model = 'inception_resnet_v2'
    img_size = 1024
    n_folds = 6
    classes = {NEGATIVE:0, TYPICAL:1, INDERTEMINATE:2, ATYPICAL:3}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 4
    num_workers = 8

# Initialize wandb
OFFLINE = False

if not OFFLINE:
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("wandb-key")
    wandb.login(key=wandb_key)

    run = wandb.init(project="siim-covid19-detection", name="multi-class-classification", mode='online', resume=True)


# In[8]:


## Check data distribution ##

train_df = pd.read_csv('../input/d/miriamassraf/siim-covid19-detection/train_df.csv')
for cls in list(Configs.classes.keys()):
    print("number of samples for class \'{}\': {}".format(cls, len(train_df[train_df['study_level']==cls])))


# In[11]:


## Split data ##

# Deal with imbalanced data by undersampling most frequent class and oversampling the others
# Split data assuring balanced distribution using StratifiedKFold
class DataFolds:
    def __init__(self, train_df, continue_train=False):
        assert Configs.n_folds > 0, "num folds must be a positive number"
        if continue_train:
            self.train_df = pd.read_csv('../input/d/miriamassraf/siim-covid19-detection/multi_class_classification/splitted_train_df.csv')
        else:
            df = train_df
            # Undersample - split frequent class and use half for each fold split
            df1, df2 = self.undersample(df, TYPICAL, 0.5)
            # Oversample - increase size of least frequent classes for each half
            df1 = self.oversample(df1, [INDERTEMINATE, ATYPICAL], [0.4, 2.0])
            df2 = self.oversample(df2, [INDERTEMINATE, ATYPICAL], [0.4, 2.0])
            # Split each df to folds (firt 0 to Configs.n_folds, second from Configs.n_folds to 2*Configs.n_folds)
            df1 = self.split_to_folds(df1)
            df2 = self.split_to_folds(df2, start_from_zero=False)
            # Create a single df with all folds
            self.train_df = df1.append(df2, ignore_index = True)

    def oversample(self, df, classes, fracs):
        for cls,frac in zip(classes, fracs):
            rows_to_add = df[df['study_level']==cls].sample(frac=frac, replace=True)
            df = df.append(rows_to_add, ignore_index = True)
        return df
    
    def undersample(self, df, cls, frac):
        freq_cls = df[df['study_level']==cls]
        half1 = freq_cls.sample(frac=frac, replace=True)
        half2 = freq_cls[~freq_cls['img_id'].isin(half1['img_id'])]
        df1 = df[df['study_level']!=cls].append(half1, ignore_index = True)
        df2 = df[df['study_level']!=cls].append(half2, ignore_index = True)
        return df1, df2
            
    def split_to_folds(self, df, start_from_zero=True):
        skf = StratifiedKFold(n_splits=Configs.n_folds//2)
        for n, (train_index, val_index) in enumerate(skf.split(X=df.index, y=df['int_label'])):
            if start_from_zero:
                df.loc[df.iloc[val_index].index, 'fold'] = int(n)
            else:
                df.loc[df.iloc[val_index].index, 'fold'] = int(n+(Configs.n_folds//2))
        return df
    
    def get_train_df(self, fold_number): 
        if fold_number >= 0 and fold_number < Configs.n_folds:
            return self.train_df[self.train_df['fold'] != fold_number]

    def get_val_df(self, fold_number):
        if fold_number >= 0 and fold_number < Configs.n_folds:
            return self.train_df[self.train_df['fold'] == fold_number]


# In[12]:


# Plot distibution
def plot_folds(data_folds):
    nrows = (Configs.n_folds//2-1)
    if (Configs.n_folds//2)%2 != 0:
        nrows += 1
    
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(30,15))
    row = 0
    for fold in range(Configs.n_folds):
        if fold%2 == 0:
            col = 0
            if fold != 0:
                row += 1
        else:
            col = 1

        labels_count = {}
        labels_count[NEGATIVE] = len(data_folds.train_df[((data_folds.train_df['fold'] == fold)&(data_folds.train_df['int_label'] == Configs.classes[NEGATIVE]))])
        labels_count[TYPICAL] = len(data_folds.train_df[((data_folds.train_df['fold'] == fold)&(data_folds.train_df['int_label'] == Configs.classes[TYPICAL]))])
        labels_count[INDERTEMINATE] = len(data_folds.train_df[((data_folds.train_df['fold'] == fold)&(data_folds.train_df['int_label'] == Configs.classes[INDERTEMINATE]))])
        labels_count[ATYPICAL] = len(data_folds.train_df[((data_folds.train_df['fold'] == fold)&(data_folds.train_df['int_label'] == Configs.classes[ATYPICAL]))])
        
        ax[row, col].bar(list(labels_count.keys()), list(labels_count.values()))

        for j, value in enumerate(labels_count.values()):
            ax[row, col].text(j, value+2, str(value), color='#267DBE', fontweight='bold')

        ax[row, col].grid(axis='y', alpha=0.75)
        ax[row, col].set_title("For fold #{}".format(fold), fontsize=15)
        ax[row, col].set_ylabel("count")


# In[13]:


data_folds = DataFolds(train_df)
data_folds.train_df.to_csv("./splitted_train_df.csv", index=False)


# In[14]:


plot_folds(data_folds)


# In[16]:


## Create custom dataset ##

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.GaussNoise(p=0.5),
                A.IAASharpen(p=0.5)
                ],p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.CLAHE(p=0.5),
            A.Resize(height=Configs.img_size, width=Configs.img_size, p=1),])

    else:
        return A.Compose([
            A.Resize(height=Configs.img_size, width=Configs.img_size, p=1),])


# In[17]:


def get_dicom_img(path):
    data_file = pydicom.dcmread(path)
    img = apply_voi_lut(data_file.pixel_array, data_file)

    if data_file.PhotometricInterpretation == "MONOCHROME1":
        img = np.amax(img) - img
    
    # Rescaling grey scale between 0-255 and convert to uint
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)

    return img


# In[18]:


class Covid19Dataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['dicom_path']
        
        img = get_dicom_img(img_path)
        label = row['int_label']
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
           
        # Normalize img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        # Convert image into a torch.Tensor
        img = torch.as_tensor(img, dtype=torch.float32)
        # Permute image to [C,H,W] from [H,W,C] and normalize
        img = img.permute(2, 0, 1)
        
        return img, label
    
    def __len__(self):
        return len(self.df)


# In[19]:


def get_dataset_fold(data_folds, fold, train=True):
    if train:
        return Covid19Dataset(data_folds.get_train_df(fold), transform=get_transforms(train))
    return Covid19Dataset(data_folds.get_val_df(fold), transform=get_transforms(train))


# In[20]:


## Train multi-class classification ##

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[27]:


class Fitter:
    def __init__(self, dir, model_name, verbose=True):
        # Create pretrained timm model by name
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=True, num_classes=len(Configs.classes))
        self.verbose = verbose
        
        self.epoch = 0 
        # Output dir
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
        self.log_path = os.path.join(self.dir, 'log.txt')
        self.best_summary_loss = 10**5

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=TrainConfigs.lr)
        self.scheduler = TrainConfigs.SchedulerClass(self.optimizer, **TrainConfigs.scheduler_params) ########
        self.log(f'Fitter prepared. Device is {Configs.device}')
        
    def fit(self, fold, train_loader, validation_loader, continue_train=False):
        self.model.to(Configs.device)
        self.log(f'Fold {fold}')
        
        if continue_train:
            self.load(f'../input/d/miriamassraf/siim-covid19-detection/multi_class_classification/{self.model_name}_fold{fold}/last-checkpoint.bin')
            
        for e in range(self.epoch, TrainConfigs.n_epochs):
            if self.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
            
            # Train one epoch
            t = time.time()
            summary_loss, summary_accuracy = self.train_one_epoch(train_loader)
            
            # Log train losses to console/log file
            self.log(f'[RESULT]: Train. Epoch: {self.epoch},\ttotal loss: {summary_loss.avg:.5f},\ttotal accuracy: {summary_accuracy.avg:.5f},\ttime: {(time.time() - t):.5f}')
            if not OFFLINE:
                # Log train losses to wandb
                run.log({f"{self.model_name}/train/total_loss_fold{fold}": summary_loss.avg})

            # Validate one epoch
            t = time.time()
            summary_loss, summary_accuracy = self.validation_one_epoch(validation_loader)
            
            # Log val losses to console/log file
            self.log(f'[RESULT]: Val. Epoch: {self.epoch},\ttotal loss: {summary_loss.avg:.5f},\ttotal accuracy: {summary_accuracy.avg:.5f},\ttime: {(time.time() - t):.5f}')
            if not OFFLINE:
                # Log val losses to wandb
                run.log({f"{self.model_name}/val/total_loss_fold{fold}": summary_loss.avg})
            
            # Save last checkpoint
            self.save(os.path.join(self.dir, 'last-checkpoint.bin'))
                        
            # Update best val losses and save best checkpoint if needed
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(os.path.join(self.dir, 'best-checkpoint.bin'))
                for path in sorted(glob(os.path.join(self.dir, 'best-checkpoint.bin')))[:-3]:
                    os.remove(path)

            self.scheduler.step(metrics=summary_loss.avg) 

            self.epoch += 1
                  
    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        summary_accuracy = AverageMeter()
        
        t = time.time()
        for step, (images, labels) in enumerate(train_loader):
            if self.verbose:
                    print(f'Train Step {step}/{len(train_loader)},\t' +                         f'total_loss: {summary_loss.avg:.5f},\t' +                         f'total_accuracy: {summary_accuracy.avg:.5f},\t' +                         f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            images = images.to(Configs.device).float()
            labels = labels.to(Configs.device).long()
            batch_size = images.shape[0]
           
            self.optimizer.zero_grad()
            
            logits = self.model(images)  
            preds = logits.argmax(dim=1 , keepdim=True)
            
            loss = TrainConfigs.loss_fn(logits, labels)
            accuracy = torchmetrics.functional.accuracy(labels, preds) ### labels, preds
            
            loss.backward()
            self.optimizer.step()
            
            summary_loss.update(loss.detach().item(), batch_size)
            summary_accuracy.update(accuracy, batch_size)
            
            del images, labels
            torch.cuda.empty_cache()

        return summary_loss, summary_accuracy
    
    def validation_one_epoch(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        summary_accuracy = AverageMeter()
        
        t = time.time()
        for step, (images, labels) in enumerate(val_loader):
            if self.verbose:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'total_loss: {summary_loss.avg:.5f}, ' + \
                        f'total_accuracy: {summary_accuracy.avg:.5f},\t' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = images.to(Configs.device).float()
                labels = labels.to(Configs.device).long()
                batch_size = images.shape[0]
    
                logits = self.model(images)
                preds = logits.argmax(dim=1, keepdim=True)
            
                loss = TrainConfigs.loss_fn(logits, labels)
                accuracy = torchmetrics.functional.accuracy(labels, preds)
                
                summary_loss.update(loss.detach().item(), batch_size)
                summary_accuracy.update(accuracy, batch_size)
                
            del images, labels
            torch.cuda.empty_cache()
            
        return summary_loss, summary_accuracy
    
    # Save checkpoint to path
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    # Load checkpoint from path
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
    
    # Log to console/log file
    def log(self, message):
        if self.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


# In[29]:


# Train configurations
class TrainConfigs:
    n_epochs = 40
    lr = 0.0001
    loss_fn = nn.CrossEntropyLoss() 
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.1,
        patience=2,
        verbose=True, 
        threshold=0.0001,
        threshold_mode='abs',
        min_lr=1e-8,
    )


# In[30]:


# Run train
def run_training(model_name, fold, train_dataset, val_dataset):
    # Create train/validation data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Configs.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=Configs.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=Configs.batch_size,
        num_workers=Configs.num_workers,
        shuffle=False,
        sampler=SequentialSampler(val_dataset),
        pin_memory=False,
    )
    
    # Create and run fitter for model
    fitter = Fitter(f'./{model_name}_fold{fold}', model_name)
    fitter.fit(fold, train_loader, val_loader, continue_train=True)


# In[ ]:


for fold in range(Configs.n_folds):
    train_dataset = get_dataset_fold(data_folds, fold)
    val_dataset = get_dataset_fold(data_folds, fold, train=False)
    run_training(Configs.model, fold, train_dataset, val_dataset)

