import torch
from torch import nn
from torch.optim import Adam

from torchvision.models import resnet50

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.utils.data import random_split
from torchvision.utils import make_grid

from torchsummary import summary
from torchmetrics import Accuracy, Precision, Recall

from matplotlib import pyplot as plt

# Load dataset
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_image_file(filename):
    return not filename.endswith('_mask.png')

train_dataset = ImageFolder('Dataset_BUSI_with_GT', transform=train_transform, is_valid_file=is_image_file)
test_dataset = ImageFolder('Dataset_BUSI_with_GT', transform=test_transform, is_valid_file=is_image_file)


print('Following classes are found in the dataset: ', train_dataset.classes)




# Check if any mask files are in the training dataset
has_mask_files = any('_mask.png' in img_path for img_path, _ in train_dataset.imgs)

if has_mask_files:
    print("Warning: Mask files are still present in the dataset!")
else:
    print("No mask files found in the dataset. All good!")



img, label = next(iter(train_dataset))
print(img.shape, label)
img = img.squeeze().permute(1, 2, 0)	

plt.imshow(img)
plt.show()

# Split dataset
train_size = int(0.7 * len(train_dataset))
test_size = len(test_dataset) - train_size
train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Visualize images of a single batch
def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        plt.show()
        break
        
show_batch(train_loader)


# Load model
# test everything on cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = len(train_dataset.dataset.classes)
print('Number of classes in the dataset: ', num_classes)

# Create a CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size = 3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(64, 128, kernel_size = 3, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size = 3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Flatten(),
            nn.Linear(256*56*56, 512),
            nn.ELU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Training the model
def train_model(model, criterion, optimizer, train_loader, device):
    model.train()
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    return loss.item()

# Metrics
metric_acc = Accuracy(task='multiclass', num_classes=num_classes, average='micro').to(device)
metric_precision = Precision(task='multiclass', num_classes=num_classes, average='micro').to(device)
metric_recall = Recall(task='multiclass', num_classes=num_classes, average='micro').to(device)

# Testing the model	
def test_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for image, labels in test_loader:
            image, labels = image.to(device), labels.to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            metric_acc(predicted, labels)
            metric_precision(predicted, labels)
            metric_recall(predicted, labels)
    
    acc = metric_acc.compute()
    prec = metric_precision.compute()
    rec = metric_recall.compute()
    print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}')



cnn_model = CNN(num_classes).to(device)
summary(cnn_model, (3, 224, 224))

resnet = resnet50(weights="IMAGENET1K_V1").to(device)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes).to(device)
summary(resnet, (3, 224, 224))


optimizer_cnn = Adam(cnn_model.parameters(), lr=0.001)
criterion_cnn = nn.CrossEntropyLoss()

optimizer_resnet = Adam(resnet.parameters(), lr=0.001)
criterion_resnet = nn.CrossEntropyLoss()

# Training loop
for epoch in range(25):
    loss = train_model(cnn_model, criterion_cnn, optimizer_cnn, train_loader, device)
    print(f'Epoch: {epoch}, CNN Loss: {loss}')

    loss = train_model(resnet, criterion_resnet, optimizer_resnet, train_loader, device)
    print(f'Epoch: {epoch}, Resnet Loss: {loss}')

#torch.save(cnn_model.state_dict(), 'Models/model_CNN.pth')
#torch.save(resnet.state_dict(), 'Models/model_Resnet.pth')

print('CNN Model:')
test_model(cnn_model, test_loader, device)

print('Resnet Model:')
test_model(resnet, test_loader, device)
