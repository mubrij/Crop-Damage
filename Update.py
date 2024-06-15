from fastai.vision.all import *
from timm import create_model
import pandas as pd
from pathlib import Path
from fastai.callback.tracker import SaveModelCallback
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from fastai.data.transforms import get_files
from fastai.callback.mixup import *
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import timm
from imblearn.over_sampling import RandomOverSampler

import torch.nn.functional as F
import torch
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler

SEED = 2023
N_FOLDS = 10
BATCH_SIZE = 40
IMGSZ = 224
EPOCHS = 100
INIT_LR = 2e-4
NUM_WORKER = 8
PATIENCE = 5
MODEL_BASE =  "vit_large_patch16_224" #'convnext_large_in22k' # vit_small_patch16_224, convnext_base_fb_in22k

Train = 'Train (27).csv'
Test = 'Test (30).csv'

set_seed(SEED, reproducible=True)

def prepare_train_data(data, kfold, image_dir):
    df = data.copy()
    df['image_id'] = df['filename'].apply(lambda x: x.split('.')[0])
    df = df.drop_duplicates(subset='image_id', keep='first')

    df['target'] = df['damage']

    df['fold'] = -1
    for i, (train_idx, val_idx) in enumerate(kfold.split(df, df['target'])):
        df.loc[val_idx, 'fold'] = i

    print(df.groupby(['fold', 'target']).size())

    df['path'] = df['filename'].apply(lambda x: f'{image_dir}/{x}')
    df['fold'] = df['fold'].astype('int')

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(df['target']), y=df['target'])
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return df, class_weights

def apply_sampling(df, fold):
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=SEED)
    _, sample_indices = undersampler.fit_resample(df[df['fold'] != fold][['path']], df[df['fold'] != fold]['target'])
    df.loc[df.index.isin(sample_indices), 'undersampled'] = True
    df['undersampled'] = df['undersampled'].fillna(False).astype(bool)

    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=SEED)
    _, sample_indices = oversampler.fit_resample(df[df['fold'] == fold][['path']], df[df['fold'] == fold]['target'])
    df.loc[df.index.isin(sample_indices), 'oversampled'] = True
    df['oversampled'] = df['oversampled'].fillna(False).astype(bool)

def cross_entropy(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()

def train_model(data):
    df = data[0].copy()
    class_weights = data[1]

    for fold in range(N_FOLDS):
        df['is_valid'] = (df['fold'] == fold)
        print(f'Training fold: {fold}')

        #apply_sampling(df, fold)  

        dls = ImageDataLoaders.from_df(
            df,
            valid_col='is_valid',
            seed=SEED,
            fn_col='path',
            label_col='target',
            label_delim=' ',
            y_block=MultiCategoryBlock,
            bs=BATCH_SIZE,
            num_workers=NUM_WORKER,
            item_tfms=Resize(IMGSZ),
            batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
            oversample_col='oversampled',
            undersample_col='undersampled'
        )


        model = create_model(f'{MODEL_BASE}', pretrained=False, num_classes=5)
        loss_func = nn.BCEWithLogitsLoss(weight=class_weights)

        learn = Learner(dls, model, loss_func=loss_func, metrics=AccumMetric(cross_entropy)).to_fp16()
        learn.fit_one_cycle(EPOCHS, INIT_LR,
                            cbs=[SaveModelCallback(), CSVLogger(append=True)])

        learn = learn.to_fp32()
        learn.save(f'__{MODEL_BASE}_fold__{fold}__', with_opt=False)

if __name__ == "__main__":
    train = pd.read_csv(Train)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    train_data, class_weights = prepare_train_data(train, skf, 'images')
    train_model((train_data, class_weights))
