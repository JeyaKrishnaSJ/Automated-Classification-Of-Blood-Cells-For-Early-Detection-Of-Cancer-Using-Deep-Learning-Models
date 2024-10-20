## BreadcrumbsAutomated-Classification-Of-Blood-Cells-For-Early-Detection-Of-Cancer-Using-Deep-Learning-Models
### Program:
```py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load the dataset
file_path = []
labels = []

# Define directories
Benign_dir = r"C:/Users/SEC/Downloads/archive/Blood cell Cancer [ALL]/Benign"
Malignant_Pre_B_dirs = r"C:/Users/SEC/Downloads/archive/Blood cell Cancer [ALL]/[Malignant] Pro-B"
Malignant_Pro_B_dirs = r"C:/Users/SEC/Downloads/archive/Blood cell Cancer [ALL]/[Malignant] Pre-B"
Malignant_early_Pre_B_dirs = r"C:/Users/SEC/Downloads/archive/Blood cell Cancer [ALL]/[Malignant] early Pre-B"

dict_list = [Benign_dir, Malignant_Pre_B_dirs, Malignant_Pro_B_dirs, Malignant_early_Pre_B_dirs]
class_labels = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']

# Collect file paths and labels
for i, dir_list in enumerate(dict_list):
    flist = os.listdir(dir_list)
    for f in flist:
        fpath = os.path.join(dir_list, f)
        file_path.append(fpath)
        labels.append(class_labels[i])

# Create a dataframe
blood_cell_df = pd.DataFrame({'filepaths': file_path, 'labels': labels})

# Train-test split
train_image, test_image = train_test_split(blood_cell_df, test_size=0.3, random_state=42)

# Image size and batch size
image_size = (224, 224)
batchsize = 8

# PyTorch custom dataset class
class BloodCellDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepaths']
        label = self.dataframe.iloc[idx]['labels']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets and data loaders
train_dataset = BloodCellDataset(train_image, transform=train_transform)
test_dataset = BloodCellDataset(test_image, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

# Model: DenseNet and ResNet using PyTorch
densenet_model = models.densenet121(pretrained=True)
resnet_model = models.resnet50(pretrained=True)

# Freeze most layers for faster training
for param in densenet_model.parameters():
    param.requires_grad = False

for param in resnet_model.parameters():
    param.requires_grad = False

# Modify both models to remove the final classification layer and replace it with Flatten
densenet_model.classifier = nn.Sequential(
    nn.Flatten(),  # Flatten the global average pooled output
)

resnet_model.fc = nn.Sequential(
    nn.Flatten(),  # Flatten the global average pooled output
)

# Feature extraction function
def extract_features(dataloader, model):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = model(images)  # Now, this should work fine
            features.append(outputs.cpu().numpy())
            labels.extend(lbls)
    
    features = np.concatenate(features)
    return features, labels

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
densenet_model.to(device)
resnet_model.to(device)

# Extract features using both models
densenet_features_train, train_labels = extract_features(train_loader, densenet_model)
densenet_features_test, test_labels = extract_features(test_loader, densenet_model)
resnet_features_train, _ = extract_features(train_loader, resnet_model)
resnet_features_test, _ = extract_features(test_loader, resnet_model)

# Combine features
train_features_combined = np.concatenate((densenet_features_train, resnet_features_train), axis=1)
test_features_combined = np.concatenate((densenet_features_test, resnet_features_test), axis=1)

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features_combined)
test_features_scaled = scaler.transform(test_features_combined)

# Convert labels to numerical format
label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
train_labels = np.array([label_to_idx[label] for label in train_labels])
test_labels = np.array([label_to_idx[label] for label in test_labels])

# Convert to PyTorch tensors
train_features_scaled = torch.tensor(train_features_scaled, dtype=torch.float32)
test_features_scaled = torch.tensor(test_features_scaled, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# PyTorch Classifier (Simple Feed-Forward Neural Network)
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model, Loss, Optimizer
classifier = Classifier(train_features_scaled.shape[1], len(class_labels))
classifier.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Train the classifier
epochs = 10
for epoch in range(epochs):
    classifier.train()
    
    optimizer.zero_grad()
    outputs = classifier(train_features_scaled.to(device))
    loss = criterion(outputs, train_labels.to(device))
    
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the classifier
classifier.eval()
with torch.no_grad():
    test_outputs = classifier(test_features_scaled.to(device))
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == test_labels.to(device)).sum().item() / len(test_labels)
    print(f'Test Accuracy: {accuracy:.4f}')

# Confusion Matrix and Classification Report
cm = confusion_matrix(test_labels.cpu(), predicted.cpu())
cr = classification_report(test_labels.cpu(), predicted.cpu(), target_names=class_labels)

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print Classification Report
print("Classification Report:")
print(cr)

```



### Output:
![image](https://github.com/user-attachments/assets/6450263d-3d41-4bd8-b735-bfbf3b0314b4)

![image](https://github.com/user-attachments/assets/a8110f1d-8fc0-419a-b80f-9edf3d80460f)

![image](https://github.com/user-attachments/assets/575382f1-5269-44d8-8aac-b91d6f1b44a0)
