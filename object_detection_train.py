import sys
sys.path.append("../input/efficientdet-pytorch-master/efficientdet-pytorch-master")
sys.path.append("../input/omegaconf")

import torch
import os
import ast
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# --- time ---
from datetime import datetime
import time
# --- images ---
import cv2
import albumentations as A
# --- data ---
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
# --- effdet ---
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict #DetBenchEval
from effdet.efficientdet import HeadNet
# --- wandb ---
import wandb
from kaggle_secrets import UserSecretsClient
# --- dicom ---
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
# --- warnings ---
import warnings
warnings.filterwarnings('ignore')


## Configurations ##

OFFLINE = False

NEGATIVE = 'negative'
TYPICAL = 'typical'
INDETERMINATE = 'indeterminate'
ATYPICAL = 'atypical'

class Configs:
    img_size = 768 # 1024 896 768 640 512
    n_folds = 6
    thing_classes = {TYPICAL:1, INDETERMINATE:2, ATYPICAL:3}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Initialize wandb
if not OFFLINE:
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("wandb-key")
    wandb.login(key=wandb_key)

    run = wandb.init(project="siim-covid19-detection", name="object_detection", resume=True, mode='online')


# Read train df
train_df = pd.read_csv('../input/siim-covid19-data/train_df.csv')


## Preprocessing and functions overriding ##

# Assert bbox boundaries to image boundaries
rows_to_change = {}
for index, row in train_df.iterrows():
    boxes = ast.literal_eval(row['pascal_voc_boxes'])
    new_boxes = []
    change = False
    for box in boxes:
        x1,y1,x2,y2 = box
        img_shape = ast.literal_eval(row['image_shape'])
        if x1<0 or y1<0 or x2>img_shape[1] or y2>img_shape[0]:
            print("For image: {}\nShape: {}\nBounding box: {}".format(row['img_id'], img_shape, box))
            change = True
        
            if x1<0:
                x1=0
            if y1<0:
                y1=0
            if x2>img_shape[1]:
                x2=img_shape[1]
            if y2>img_shape[0]:
                y2=img_shape[0]

        new_boxes.append([x1,y1,x2,y2])
        
    if change:
        rows_to_change[row['img_id']] = new_boxes
        
for img_id, new_boxes in rows_to_change.items():
    train_df.loc[train_df['img_id'] == img_id, 'pascal_voc_boxes'] = str(new_boxes)


# Override checkbox function and assert bbox boundaries within normalized [0-1]
def new_check_bbox(bbox):
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    # New block 
    bbox=list(bbox)
    for i in range(4):
        if (bbox[i]<0) :
            bbox[i]=0
        elif (bbox[i]>1) :
            bbox[i]=1
    bbox=tuple(bbox)
    # End new block 
    
    for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
        if not 0 <= value <= 1:
            raise ValueError(
                "Expected {name} for bbox {bbox} "
                "to be in the range [0.0, 1.0], got {value}.".format(bbox=bbox, name=name, value=value)
            )
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max <= x_min:
        raise ValueError("x_max is less than or equal to x_min for bbox {bbox}.".format(bbox=bbox))
    if y_max <= y_min:
        raise ValueError("y_max is less than or equal to y_min for bbox {bbox}.".format(bbox=bbox))

A.augmentations.bbox_utils.check_bbox = new_check_bbox


## Check data distribution ##

num_negatives = len(train_df[train_df['study_level'] == 'negative']) 
num_typicals = len(train_df[train_df['study_level'] == 'typical']) 
num_atypical = len(train_df[train_df['study_level'] == 'atypical']) 
num_indeterminates = len(train_df[train_df['study_level'] == 'indeterminate'])

counts = {'negative': num_negatives, 'typical': num_typicals, 'atypical': num_atypical, 'indeterminate': num_indeterminates}
print(counts)

total = counts['negative'] + counts['typical'] + counts['atypical'] + counts['indeterminate']

print('Total : ', total)
print('%negatives = {:.2f}'.format((counts['negative']/total) * 100))
print('%typicals = {:.2f}'.format((counts['typical']/total) * 100))
print('%atypicals = {:.2f}'.format((counts['atypical']/total) * 100))
print('%indeterminates = {:.2f}'.format((counts['indeterminate']/total) * 100))


## Split data ##

# Deal with imbalanced data by undersampling most frequent class and oversampling the others
# Split data assuring balanced distribution using StratifiedKFold
class DataFolds:
    def __init__(self, train_df, continue_train=False, debug=False):
        assert Configs.n_folds > 0, "num folds must be a positive number"
        if continue_train:
            self.train_df = pd.read_csv('../input/siim-covid19-data/object_detection/splitted_train_df.csv')
        else:
            df = train_df
            # Undersample - split frequent class and use half for each fold split
            df1, df2 = self.undersample(df, TYPICAL, 0.5)
            # Oversample - increase size of least frequent classes for each half
            df1 = self.oversample(df1, [INDETERMINATE, ATYPICAL], [0.4, 2.0])
            df2 = self.oversample(df2, [INDETERMINATE, ATYPICAL], [0.4, 2.0])
            # Split each df to folds (firt 0 to Configs.n_folds, second from Configs.n_folds to 2*Configs.n_folds)
            df1 = self.split_to_folds(df1)
            df2 = self.split_to_folds(df2, start_from_zero=False)
            # Create a single df with all folds
            self.train_df = df1.append(df2, ignore_index = True)
        
        if debug:
            self.train_df = self.train_df.sample(frac=0.02)
    
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

data_folds = DataFolds(train_df)
data_folds.train_df.to_csv("splitted_train_df.csv", index=False)


# Plot distibution
def plot_folds(data_folds):
    nrows = Configs.n_folds//2
    if Configs.n_folds%2 != 0:
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
        labels_count[TYPICAL] = len(data_folds.train_df[((data_folds.train_df['fold'] == fold)&(data_folds.train_df['study_level'] == TYPICAL))])
        labels_count[ATYPICAL] = len(data_folds.train_df[((data_folds.train_df['fold'] == fold)&(data_folds.train_df['study_level'] == ATYPICAL))])
        labels_count[INDETERMINATE] = len(data_folds.train_df[((data_folds.train_df['fold'] == fold)&(data_folds.train_df['study_level'] == INDETERMINATE))])

        ax[row, col].bar(list(labels_count.keys()), list(labels_count.values()))

        for j, value in enumerate(labels_count.values()):
            ax[row, col].text(j, value+2, str(value), color='#267DBE', fontweight='bold')

        ax[row, col].grid(axis='y', alpha=0.75)
        ax[row, col].set_title("For fold #{}".format(fold), fontsize=15)
        ax[row, col].set_ylabel("count")

plot_folds(data_folds)


## Create custom dataset ##

def get_transforms(train: bool=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10),
            A.OneOf([
                A.HueSaturationValue(), 
                A.RandomBrightnessContrast(),
                A.CLAHE(p=0.6),
            ], p=0.4),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.GaussNoise(p=0.5),
                A.Sharpen(p=0.5)
                ],p=0.4),
            A.Resize(height=Configs.img_size, width=Configs.img_size, p=1),], 
            bbox_params=A.BboxParams(format='pascal_voc',
                                     min_area=0, 
                                     min_visibility=0,
                                     label_fields=['labels'])
        )

    else:
        # For validation only resize image
        return A.Compose([
            A.Resize(height=Configs.img_size, width=Configs.img_size, p=1),], 
            bbox_params=A.BboxParams(format='pascal_voc',
                                     min_area=0, 
                                     min_visibility=0,
                                     label_fields=['labels'])
        )


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


class Covid19Dataset(Dataset):
    def __init__(self, df, train=True, transform=None):
        super().__init__()
        self.df = df
        self.train = train
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['dicom_path']
        
        img = get_dicom_img(img_path)
        bboxes = ast.literal_eval(row['pascal_voc_boxes'])
        if row['num_boxes'] > 0:
            labels = [row['int_label']]*row['num_boxes']
        else:
            labels = [row['int_label']]
            
        if self.transform:
            transformed = self.transform(**{'image': img, 'bboxes': bboxes, 'labels': labels})
            img = transformed['image']
            transformed_bboxes = transformed['bboxes']        
            
            if row['num_boxes'] > 0:
                bboxes = transformed_bboxes

        # Normalize img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        # convert everything into a torch.Tensor
        img = torch.as_tensor(img, dtype=torch.float32)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        idx = torch.tensor([idx])
        
        bboxes[:,[0,1,2,3]] = bboxes[:,[1,0,3,2]]  #yxyx: be warning
        
        # Targets for object detection
        target = {}
        target['bbox'] = bboxes
        target['cls'] = labels
        target['img_id'] = idx         
        # Permute image to [C,H,W] from [H,W,C] and normalize
        img = img.permute(2, 0, 1)
        
        return img, target, row['img_id']

    def __len__(self):
        return len(self.df)


# Get train/validation folds
def get_dataset_fold(data_folds, fold,train=True):
    if train:
        return Covid19Dataset(data_folds.get_train_df(fold), train=True, transform=get_transforms(train))
    return Covid19Dataset(data_folds.get_val_df(fold), train=False, transform=get_transforms(train))


## Competition metric - PASCAL VOC 2010 mean average precision (mAP) at IoU > 0.5 ##

# https://www.kaggle.com/chenyc15/mean-average-precision-metric 
# For format pascal_voc: [xmin, ymin, xmax, ymax]
def calculate_iou(gt, pred):
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pred: (np.ndarray[Union[int, float]]) coordinates of the prdected box
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    pred_x1, pred_y1, pred_x2, pred_y2 = pred

    # Calculate overlap area
    dx = min(gt_x2, pred_x2) - max(gt_x1, pred_x1) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt_y2, pred_y2) - max(gt_y1, pred_y1) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_x1 + 1) +
            (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1) -
            overlap_area
    )

    return overlap_area / union_area

def find_best_match(gts, pred, pred_idx, threshold = 0.5, ious=None):
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        if gts[gt_idx][0] < 0:
            # Already matched GT-box (set to -1)
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

def calculate_image_precision_recall(gts, preds, threshold = 0.5):
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
    Return:
        (float) Precision
    """
    ious = np.ones((len(gts), len(preds))) * -1
    
    n = len(preds)
    tp = 0
    fp = 0
    
    # For pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):
        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1
    
    precision = tp / (tp + fp)
    recall = tp / len(gts)
    
    return precision, recall

# https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/evaluation/metrics.py
def compute_average_precision(precision: np.ndarray, recall: np.ndarray):
    """Compute Average Precision according to the definition in VOCdevkit.
    Precision is modified to ensure that it does not decrease as recall
    decrease.
    Args:
        precision: A float [N, 1] numpy array of precisions
        recall: A float [N, 1] numpy array of recalls
    Raises:
        ValueError: if the input is not of the correct format
    Returns:
        average_precison: The area under the precision recall curve. NaN if
            precision and recall are None.
    """
    if precision is None:
        if recall is not None:
            raise ValueError("If precision is None, recall must also be None")
        return np.NAN

    if len(precision) != len(recall):
        raise ValueError("precision and recall must be of the same size.")

    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError("Precision must be in the range of [0, 1].")
    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError("recall must be in the range of [0, 1].")
    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError("recall must be a non-decreasing array")
    
    # Recall sorted in increasing order with values 0-1
    recall = np.concatenate([[0], recall, [1]])
    # precision as well, but we need max precision so we don't concatenate with 1 at the end
    precision = np.concatenate([[0], precision, [0]])  

    # "Smooth" curves to rectangles by getting the max value
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])
    
    # Calculate the sum of rectangle areas
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision


## Train object detection ##

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


class Fitter:
    def __init__(self, dir):
        self.epoch = 0
        # Create EfficientDet model
        self.model = get_net()
        self.device = Configs.device
        # Output dir
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
        self.log_path = os.path.join(self.dir, 'log.txt')
        self.best_summary_loss = 10**5

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=TrainGlobalConfig.lr)
        self.scheduler = TrainGlobalConfig.SchedulerClass(self.optimizer, **TrainGlobalConfig.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')
        
    def fit(self, fold, train_loader, validation_loader, continue_train=False):
        if continue_train:
            path = f'../input/siim-covid19-data/object_detection/{self.dir}/last-checkpoint.bin'
            self.load(path)
        else:
            self.log(f"Fold {fold}")
            
        while self.epoch < TrainGlobalConfig.n_epochs:
            if TrainGlobalConfig.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
            
            
            # Train one epoch
            t = time.time()
            summary_loss, summary_box_loss, summary_class_loss = self.train_one_epoch(train_loader)
            # Log train losses to console/log file
            self.log(f'[RESULT]: Train. Epoch: {self.epoch},\t' +                      f'total loss: {summary_loss.avg:.5f},\t' +                      f'loss_cls: {summary_class_loss.avg:.5f},\t' +                      f'loss_box_reg: {summary_box_loss.avg:.5f},\t' +                      f'time: {(time.time() - t):.5f}')
            # Log train losses to wandb
            if not OFFLINE:
                run.log({f"train/total_loss_fold{fold}": summary_loss.avg})
                run.log({f"train/loss_box_reg_fold{fold}": summary_box_loss.avg})
                run.log({f"train/loss_cls_fold{fold}": summary_class_loss.avg})

            # Save last checkpoint
            self.save(f'./{self.dir}/last-checkpoint.bin')
            
            # Validate one epoch
            t = time.time()
            summary_loss, summary_box_loss, summary_class_loss, mAP = self.validation_one_epoch(validation_loader)
            
            # Log val losses to console/log file
            if mAP is not None:
                self.log(f'[RESULT]: Val. Epoch: {self.epoch},\ttotal loss: {summary_loss.avg:.5f},\t' +                      f'loss_cls: {summary_class_loss.avg:.5f},\t' +                      f'loss_box_reg: {summary_box_loss.avg:.5f},\t' +                      f'time: {(time.time() - t):.5f}\n' +                      '-'*100 +                      f'\nmAP@IoU=0.5: {mAP},\n' +                      '-'*100)
            else:
                 self.log(f'[RESULT]: Val. Epoch: {self.epoch},\ttotal loss: {summary_loss.avg:.5f},\t' +                      f'loss_cls: {summary_class_loss.avg:.5f},\t' +                      f'loss_box_reg: {summary_box_loss.avg:.5f},\t' +                      f'time: {(time.time() - t):.5f}')   

            # Log val losses to wandb
            if not OFFLINE:
                run.log({f"val/total_loss_fold{fold}": summary_loss.avg})
                run.log({f"val/loss_box_reg_fold{fold}": summary_box_loss.avg})
                run.log({f"val/loss_cls_fold{fold}": summary_class_loss.avg})
                if (self.epoch+1)%10 == 0 and self.epoch != 0:
                    run.log({f"mAP_fold{fold}/IoU=0.5": mAP})
                
            # Update best val losses and save best checkpoint if needed
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'./{self.dir}/best-checkpoint.bin')
            
            # Perform scheduler step
            if TrainGlobalConfig.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        summary_box_loss = AverageMeter()
        summary_class_loss = AverageMeter()
        
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if TrainGlobalConfig.verbose:
                print(f'Train Step {step}/{len(train_loader)},\t' +                     f'total_loss: {summary_loss.avg:.5f},\t' +                     f'loss_cls: {summary_class_loss.avg:.5f},\t' +                     f'loss_box_reg: {summary_box_loss.avg:.5f},\t' +                     f'time: {(time.time() - t):.5f}', end='\r'
                )
                    
            images = torch.stack(images)
            batch_size = images.shape[0]
            
            images = images.to(self.device).float()
            boxes = [target['bbox'].to(self.device).float() for target in targets]
            classes = [target['cls'].to(self.device).float() for target in targets]
            targets = {'bbox':boxes, 'cls':classes}
           
            self.optimizer.zero_grad()
            output  = self.model(images, targets)
            loss, class_loss, box_loss = output['loss'], output['class_loss'], output['box_loss']
            
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)
            summary_box_loss.update(box_loss.detach().item(), batch_size)
            summary_class_loss.update(class_loss.detach().item(), batch_size)
            
            self.optimizer.step()
            del images, targets, image_ids
            
        return summary_loss, summary_box_loss, summary_class_loss
    
    def validation_one_epoch(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        summary_box_loss = AverageMeter()
        summary_class_loss = AverageMeter()
        precisions = []
        recalls = []
        
        t = time.time()
            
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if TrainGlobalConfig.verbose:
                print(
                    f'Val Step {step}/{len(val_loader)}, ' + \
                    f'summary_loss: {summary_loss.avg:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}', end='\r'
                )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                
                images = images.to(self.device).float()
                boxes = [target['bbox'].to(self.device).float() for target in targets]
                classes = [target['cls'].to(self.device).float() for target in targets]
                scales = torch.tensor([1. for target in targets]).to(self.device).float()
                sizes = torch.tensor([(Configs.img_size, Configs.img_size) for target in targets]).to(self.device).float()
            
                targets = {'bbox':boxes, 'cls':classes, 'img_scale':scales, 'img_size':sizes}

                output  = self.model(images, targets)
                loss, class_loss, box_loss = output['loss'], output['class_loss'], output['box_loss']
                
                summary_loss.update(loss.detach().item(), batch_size)
                summary_box_loss.update(box_loss.detach().item(), batch_size)
                summary_class_loss.update(class_loss.detach().item(), batch_size)
                
                # Evaluate mAP every 5 epochs (4, 9, 14, 19, 24, 29)
                if (self.epoch+1)%5 == 0: 
                    # Get eval model
                    eval_model = get_net(train=False)
                    eval_model.eval();
                    # Load current model weights
                    state_dict = self.model.state_dict()
                    eval_model.load_state_dict(state_dict)

                    preds = eval_model(images) 
                    for i, gt_boxes in enumerate(boxes):               
                        precision, recall = self.calc_precision_recall(preds, gt_boxes, i)
                        precisions.append(precision)
                        recalls.append(recall)
                    
            del images, targets, image_ids
            
        mAP = None
        if (self.epoch+1)%5 == 0:
            mAP = self.calc_mAP(precisions, recalls)
        
        return summary_loss, summary_box_loss, summary_class_loss, mAP
    
    def calc_precision_recall(self, preds, gt_boxes, i):
        pred_boxes = preds[i].detach().cpu().numpy()[:,:4]
        pred_scores = preds[i].detach().cpu().numpy()[:,4]
        pred_labels = preds[i].detach().cpu().numpy()[:, 5]

        # Sort predictions by score
        preds_sorted_idx = np.argsort(pred_scores)[::-1]
        preds_sorted_boxes = pred_boxes[preds_sorted_idx]

        return calculate_image_precision_recall(gt_boxes.detach().cpu().numpy(), preds_sorted_boxes)
                        
    def calc_mAP(self, precisions, recalls):
        # Sort by recall (increasing order)
        recalls = np.array(recalls)
        precisions = np.array(precisions)
        sorted_idx = np.argsort(recalls)
        recalls = recalls[sorted_idx]
        precisions = precisions[sorted_idx]
        return compute_average_precision(precisions, recalls)
            
    # Save checkpoint to given path
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    # Load checkpoint from given path
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        #self.epoch = checkpoint['epoch'] + 1
        self.epoch = checkpoint['epoch']
    
    # Log to console and log file
    def log(self, message):
        if TrainGlobalConfig.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


# Create Effdet model
def get_net(architecture='tf_efficientdet_d7', train=True):
    config = get_efficientdet_config(architecture)
    config.num_classes = len(Configs.thing_classes)
    config.image_size = (Configs.img_size,Configs.img_size)
    config.gamma = 2
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    if train:
        print(config)
        return DetBenchTrain(net, config).to(Configs.device)

    else:
        return DetBenchPredict(net).to(Configs.device)


# Train configurations
class TrainGlobalConfig:
    n_epochs = 40
    num_workers = 8
    batch_size = 2
    lr = 0.0001
    verbose = True
    validation_scheduler = True  

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


# Run train
def collate_fn(batch):
    return tuple(zip(*batch))

def run_training(fold, train_dataset, val_dataset, continue_train=False):
    # Create tain/validation data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(val_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )
    
    # Create fitter for model
    fitter = Fitter(f'tf_efficientdet_d7_fold{fold}')
    # Run train by calling fit function
    fitter.fit(fold, train_loader, val_loader, continue_train)

for fold in range(Configs.n_folds):
    train_dataset = get_dataset_fold(data_folds, fold)
    val_dataset = get_dataset_fold(data_folds, fold, train=False)
    run_training(fold, train_dataset, val_dataset)

