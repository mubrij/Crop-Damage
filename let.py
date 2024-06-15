import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from torchvision.utils import make_grid
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.utils import resample
import random
import torch
import torch.nn as nn
from torchvision import models


from torch.autograd import Variable
import argparse
import torch.backends.cudnn as cudnn

import cv2
from efficientnet_pytorch import EfficientNet
from torchvision.models import googlenet
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
#sudo docker run -it nvcr.io/nvidia/pytorch:23.07-py3
from tqdm.auto import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

np.random.seed(10)
torch.manual_seed(10)
random.seed(10)

BATCH_SIZE = 50
IMAGE_SIZE = (225, 225)
TRAIN_CSV = 'Train (27).csv'
TEST_CSV = 'Test (30).csv'

damage_definitions = {
    'DR': 'Drought',
    'G': 'Good (growth)',
    'ND': 'Nutrient Deficient',
    'WD': 'Weed',
    'other': 'Disease, Pest, Wind'
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class CustomDataset(Dataset):
  def __init__(self, csv_file, image_folder='images', transform=None, balance_dataset=False):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.balance_dataset = balance_dataset

        if self.balance_dataset:
            self.undersampler = RandomUnderSampler(sampling_strategy='not minority')
            self.oversampler = RandomOverSampler(sampling_strategy='minority')
            X, y = self.data.drop('damage', axis=1), self.data['damage']
            X_resampled, y_resampled = self.undersampler.fit_resample(X, y)
            X_resampled, y_resampled = self.oversampler.fit_resample(X_resampled, y_resampled)

            # Update data with resampled data
            self.data = pd.concat([X_resampled, y_resampled], axis=1)
            
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    img_name = self.data.loc[idx,'filename']
    img_path = f"{self.image_folder}/{img_name}"
    image = Image.open(img_path).convert('RGB')
    if self.transform:
      image = self.transform(image)
    if 'damage' in self.data.columns:
      damage = self.data.loc[idx, ['damage']]
      ohe = pd.get_dummies(self.data['damage'])
      df_ohe = pd.concat([self.data,ohe],axis=1)
      df_ohe = df_ohe.drop('damage',axis=1)
      labels = df_ohe.loc[idx, ['DR','G','ND','WD','other']]
      labels = torch.tensor(labels.values.astype('int8'),dtype=torch.float)
      return image, labels
    return image

  def plot_damage(self):
    try:
      counts = self.data.damage.value_counts()
      plt.bar(counts.index, counts.values)
      for i, value in enumerate(counts.values):
        plt.text(i, value, str(value), ha='center', va='bottom')
      plt.xlabel('Damage Type')
      plt.ylabel('Count')
      plt.title('Distribution of Damage Types')
      plt.show()
    except Exception as e:
      print('It is test dataset and it doesn\'t have damage types.',e)
      

def custom_opencv_transform(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * 1.8  # Increase saturation
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    smoothed_image = cv2.GaussianBlur(saturated_image, (5, 5), 0)

    gray_smoothed = cv2.medianBlur(gray_image, 5)
    edges = cv2.adaptiveThreshold(gray_smoothed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    pencil_sketch_image = cv2.bitwise_and(smoothed_image, color_edges)

    return pencil_sketch_image

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
    transforms.GaussianBlur(kernel_size=3),
    transforms.Lambda(lambda img: custom_opencv_transform(np.array(img))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Lambda(lambda img: custom_opencv_transform(np.array(img))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = CustomDataset(TRAIN_CSV, transform=transform)
test_data = CustomDataset(TEST_CSV, transform=transform_test)

TRAIN_SIZE = int(0.8*len(train_data))
VAL_SIZE = len(train_data) - TRAIN_SIZE
train_set, val_set = torch.utils.data.random_split(train_data,[TRAIN_SIZE,VAL_SIZE])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class ResNet101(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet101, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        in_features = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet101(x)
        return x

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=5, efficientnet_type='b0'):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(f'efficientnet-{efficientnet_type}')
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        return x

class GoogLeNetModel(nn.Module):
    def __init__(self, num_classes=5):
        super(GoogLeNetModel, self).__init__()
        self.googlenet = googlenet(pretrained=True)
        in_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.googlenet(x)

efficientnet_model = EfficientNetModel(efficientnet_type='b0').to(device)
googlenet_model = GoogLeNetModel().to(device)
resnet101= ResNet101().to(device)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = 0.01
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(model, train_loader, val_loader, epochs, device):
    train_losses = []
    val_losses = []
    accuracies = []
    
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_loader.dataset)):
        print(f"\nFold {fold + 1}/{num_folds}")
        
        train_subset = torch.utils.data.Subset(train_loader.dataset, train_indices)
        val_subset = torch.utils.data.Subset(train_loader.dataset, val_indices)

        train_loader_fold = torch.utils.data.DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
        val_loader_fold = torch.utils.data.DataLoader(val_subset, batch_size=val_loader.batch_size, shuffle=False)

        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.5)
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            adjust_learning_rate(optimizer, epoch)

            for _, data in enumerate(train_loader_fold):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            model.eval()
            val_loss = 0
            all_predicted = []  # Separate list for predicted values
            true_labels = []    # Separate list for true labels
            with torch.no_grad():
                for data in val_loader_fold:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    _, predicted = torch.max(outputs, 1)
                    all_predicted.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())  # Store true labels

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            accuracy = accuracy_score(all_predicted, true_labels)  # Use true_labels for accuracy calculation

            train_losses.append(total_loss / len(train_loader_fold))
            val_losses.append(val_loss / len(val_loader_fold))
            accuracies.append(accuracy)

            print(f'Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader_fold)} | Validation Loss: {val_loss / len(val_loader_fold)} | Accuracy: {accuracy:.4f} ')

        scheduler.step()

    print('Training finished.')

    return train_losses, val_losses, accuracies

def test(model):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)

if __name__ == "__main__":
    train_losses, epoch_train_losses, val_losses, accuracies = train(model=efficientnet_model, train_loader=train_loader, val_loader=val_loader, epochs=2, device=device)
    predicted = test(resnet101)
    
    checkpoint = {"state_dict":efficientnet_model.state_dict()}
    torch.save(checkpoint,"efficientnet_model.pt")
    
    
    
    
    
    
    
    
    
    
    
    
    