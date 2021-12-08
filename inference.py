import sys
sys.path.append("../input/efficientdet-pytorch-master/efficientdet-pytorch-master")
sys.path.append('../input/imagemodels/pytorch-image-models-master')
sys.path.append("../input/weightedboxesfusion")
sys.path.append("../input/omegaconf")

import torch
from torch import nn
import timm
import numpy as np
import pandas as pd
import gc
import os
import ast
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
# --- effdet ---
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
# --- ensemble boxes ---
import ensemble_boxes
from ensemble_boxes import *
# --- data ---
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler
# --- images ---
import albumentations as A
import cv2
# --- wandb ---
import wandb
from kaggle_secrets import UserSecretsClient
# --- dicom ---
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

## Configurations ##
OFFLINE = True

NONE = 'none'
OPACITY = 'opacity'

NEGATIVE = 'negative'
TYPICAL = 'typical'
INDERTEMINATE = 'indeterminate'
ATYPICAL = 'atypical'

class Configs:
    n_folds = 6
    img_size_classification = 1024
    img_size_detection = 768
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 2
    num_workers = 8
    classes = {0:NEGATIVE, 1:TYPICAL, 2:INDERTEMINATE, 3:ATYPICAL}


# Initialize wandb
if not OFFLINE:
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("wandb-key")
    wandb.login(key=wandb_key)

    run = wandb.init(project="siim-covid19-detection", name="inference-trial", mode='online')


## Create custom dataset ##

def get_img_id(path):
    # dicom path of format '../input/siim-covid19-detection/test/study_id/dir/image_id.dcm'
    return path.split('/')[-1].split('.')[0] # extract img_id from path


def get_study_id(path):
    # dicom path of format '../input/siim-covid19-detection/test/study_id/dir/dicom_image'
    return path.split('/')[-3]


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


def get_test_transforms(img_size):
    return A.Compose([A.Resize(height=img_size, width=img_size, p=1.0),], p=1.0)


class Covid19TestDataset(Dataset):
    def __init__(self, dicom_paths, transform=None):
        super().__init__()
        self.paths = dicom_paths
        self.transform = transform

    def __getitem__(self, idx):
        img = get_dicom_img(self.paths[idx])
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
           
        # normalize img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        # convert image into a torch.Tensor
        img = torch.as_tensor(img, dtype=torch.float32)        
        # permute image to [C,H,W] from [H,W,C] and normalize
        img = img.permute(2, 0, 1)
        
        return img, self.paths[idx]
    
    def __len__(self):
        return len(self.paths)


## Object detection ##

## Get EfficientDet trained models ##
def get_model(model_name):
    config = get_efficientdet_config(model_name)
    config.num_classes = len(Configs.classes) - 1
    config.image_size = (Configs.img_size_detection, Configs.img_size_detection)
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    model = DetBenchPredict(net)
    model.eval();
    
    return model


def load_obj_detection_model(model_name, fold, checkpoint_path):
    model = get_model(model_name)
    checkpoint = torch.load(checkpoint_path)['model_state_dict']
    model.load_state_dict(checkpoint)
    
    return model


models = []
model_name = 'tf_efficientdet_d7'
for fold in range(Configs.n_folds): 
    models.append(load_obj_detection_model(model_name, fold, f'../input/siim-covid19-data/object_detection/{model_name}_fold{fold}/best-checkpoint.bin'))


def log_prediction(image, image_id, boxes, labels):
    new_image = image.permute(1, 2, 0).numpy().copy() 
    new_image = (new_image*255).astype(np.uint8)
    image_size = max(image.shape)
    caption = []
    for label in labels:
        caption.append(Configs.study_level[label])
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(new_image, (int(x), (int(y))), (int(x+w),  int(y+h)), (255,0,0), image_size//200)
        
    wandb.log({'predictions/'+image_id: [wandb.Image(new_image, caption=', '.join(caption))]})


## Ensemble object detection models ##

# Weighted Boxe Fusion for ensembling boxes from object detection models
def run_wbf(predictions, img_idx, iou_thr=0.65, skip_box_thr=0.4, weights=None):
    # prediction[img_idx] is the prediction of each model for the same image (the image in img_idx)
    # weighted_boxes_fusion function received boxes with values in range [0-1]
    # normalize by dividing over Configs.img_size-1 (includes 0, therefore -1)
    boxes = [(prediction[img_idx]['boxes']/(Configs.img_size_detection-1)).tolist() for prediction in predictions]
    scores = [prediction[img_idx]['scores'].tolist() for prediction in predictions]
    labels = [prediction[img_idx]['labels'].astype(int).tolist() for prediction in predictions]
    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # scale back boxes by multiplying with Configs.img_size-1
    boxes = boxes*(Configs.img_size_detection-1)
    return boxes, scores, labels

# Batch prediction
def batch_predictions(images, score_threshold=0.4):
    with torch.no_grad():
        predictions = []
        images = torch.stack(images)
        images = images.to(Configs.device).float()
     
        # for each model get predictions for each image in batch
        for model in models:  
            model = model.to(Configs.device)
            model_predictions = []
            preds = model(images) 
        
            for i, image in enumerate(images):
                pred_boxes = preds[i].detach().cpu().numpy()[:,:4]
                pred_scores = preds[i].detach().cpu().numpy()[:,4]
                pred_labels = preds[i].detach().cpu().numpy()[:,5]
                indexes = np.where(pred_scores > score_threshold)[0]
                
                model_predictions.append({
                'boxes': pred_boxes[indexes],
                'scores': pred_scores[indexes],
                'labels': pred_labels[indexes]
                })
                
            # append model prediction to all predictions
            predictions.append(model_predictions) 
            del model
        del images
            
    return predictions

# Rescale boxes 
def get_boxes(image, boxes):
    # get ratio to scale boxes
    new_image = image.permute(1, 2, 0).numpy().copy() 
    height, width = new_image.shape[0], new_image.shape[1]
    h_ratio = height / Configs.img_size_detection
    w_ratio = width / Configs.img_size_detection
    
    # scale and convert from xywh to xyxy
    new_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        new_boxes.append(np.array([int(x1*w_ratio),int(y1*h_ratio),int(x2*w_ratio),int(y2*h_ratio)]))
        
    return np.array(new_boxes)    
            
def ensemble_predictions(test_loader):
    # returns a df with img_id, study_id and predicted values (labels, scores and boxes) for each image
    img_ids = []
    stdy_ids = []
    img_paths = []
    pred_boxes =[]
    pred_scores = []
    pred_labels = []
        
    for images, images_path in tqdm(test_loader):
        # get batch predictions of all models
        # predictions[0] = batch predictions for model0, predictions[1] = batch predictions for model1 and etc...
        predictions = batch_predictions(images)
        for i, (image, image_path) in enumerate(zip(images, images_path)):
            # for each image get ensembled prediction by using wbf
            boxes, scores, labels = run_wbf(predictions, img_idx=i)
            if not OFFLINE:
                log_prediction(image, image_id, boxes, labels)
            new_boxes = get_boxes(image, boxes)
            
            img_ids.append(get_img_id(image_path))
            stdy_ids.append(get_study_id(image_path))
            img_paths.append(image_path)
            pred_boxes.append(new_boxes)
            pred_scores.append(scores)
            pred_labels.append(labels.astype(int))

    return pd.DataFrame({ 'img_path': img_paths, 'img_id':img_ids, 'study_id':stdy_ids, 'labels':pred_labels, 'scores':pred_scores, 'boxes':pred_boxes})


## Create custom test dataset and data loader ##

# Load all paths of dicom test images
paths = []
for dirname, _, filenames in os.walk('../input/siim-covid19-detection/test'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))

test_dataset = Covid19TestDataset(paths, transform=get_test_transforms(Configs.img_size_detection))


def collate_fn(batch):
    return tuple(zip(*batch))


test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Configs.batch_size,
        sampler=SequentialSampler(test_dataset),
        shuffle=False,
        pin_memory=False,
        num_workers=Configs.num_workers,
        collate_fn=collate_fn,
    )

## Predict with ensembled object detection models ##

torch.cuda.empty_cache()
detection_df = ensemble_predictions(test_loader)
detection_df.head()


## Multi-class classification ##

## Get Inception-ResNet-V2 trained models ##
def get_model(model_name, path):
    model = timm.create_model(model_name, pretrained = False, num_classes=len(Configs.classes))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

models = []
model_name = 'inception_resnet_v2'
for fold in range(Configs.n_folds): 
    models.append(get_model(model_name, f'../input/siim-covid19-data/multi_class_classification/{model_name}_fold{fold}/best-checkpoint.bin'))


## Ensemble multi-class classification models ##
def ensemble_predictions(test_loader):
    # returns a df with img_id, study_id and predicted values (labels, scores and boxes) for each image
    img_paths = []
    pred_labels = []
    
    for images, images_path in tqdm(test_loader):
        # get batch predictions of all models
        with torch.no_grad():
            logits = []
            images = images.float().to(Configs.device)

            # for each model get predictions for each image in batch
            for model in models: 
                model = model.to(Configs.device)
                logits.append(model(images).detach().cpu().numpy())
                
            preds =  np.mean(logits, axis=0)
            pred_labels += np.argmax(preds, axis=1).tolist()
            img_paths += list(images_path)
            del images
            
    return pd.DataFrame({'img_path':img_paths, 'label':pred_labels})


## Create custom test dataset and data loader ##

# Get path only of images without object detection
paths =[]
for index, row in detection_df.iterrows():
    if len(row['labels']) == 0:
        paths.append(row['img_path'])

test_dataset = Covid19TestDataset(paths, transform=get_test_transforms(Configs.img_size_classification))


from torch.utils.data.sampler import SequentialSampler
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=Configs.batch_size,
    num_workers=Configs.num_workers,
    shuffle=False,
    pin_memory=False,
)


## Predict with ensembled classification models ##

classification_df = ensemble_predictions(test_loader)
classification_df.head()


# Append img_id and study_id classification predictions
img_ids = []
study_ids = []

for index, row in classification_df.iterrows():
    img_ids.append(get_img_id(row['img_path']))
    study_ids.append(get_study_id(row['img_path']))
    
classification_df['img_id'] = img_ids
classification_df['study_id'] = study_ids


## Create predictions csv file for submission ##

def create_image_prediction_string(detection_row):
    pred_strings = []
    if len(detection_row['labels'].values[0]) > 0:
        # if there are predicted labels
        for score,box in zip(detection_row['scores'].values[0], detection_row['boxes'].values[0]):
            x1,y1,x2,y2 = box
            pred_strings.append("opacity {:.1f} {} {} {} {}".format(score, x1, y1, x2, y2))
        return ' '.join(pred_strings)
    else:
        return 'none 1 0 0 1 1'

def create_study_prediction_string(detection_rows, classification_rows):
    pred_strings = []
    for index, row in classification_rows.iterrows():
        label = Configs.classes[row['label']]
        pred_strings.append('{} 1 0 0 1 1'.format(label))
        
    for index, row in detection_rows.iterrows():
        if len(row['labels']) > 0:
            labels = []
            for label in row['labels']:
                if label not in labels:
                    labels.append(label)
            for label in labels:
                pred_strings.append('{} 1 0 0 1 1'.format(Configs.classes[label]))
    return ' '.join(pred_strings)


# Create prediction string for each id of sample submission
def create_prediction_string(sample_submission):
    ids = []
    preds = []
    
    for index, row in sample_submission.iterrows():
        id = row['id']
        ids.append(id)
        if id.endswith('_image'):
            id = id.split('_')[0]
            preds.append(create_image_prediction_string(detection_df[detection_df['img_id'] == id]))
        else:
            id = id.split('_')[0]
            preds.append(create_study_prediction_string(detection_df[detection_df['study_id'] == id], classification_df[classification_df['study_id'] == id]))
    
    return pd.DataFrame({'id':ids, 'PredictionString':preds})

submission = create_prediction_string(pd.read_csv('../input/siim-covid19-detection/sample_submission.csv'))
submission.head()

submission.to_csv('./submission.csv', index=False)

