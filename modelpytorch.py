import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time
import os
import copy
import pathlib


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='')    
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_dir', type=str, default='')

    args = parser.parse_args()

    cudnn.benchmark = True

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
    }

    image_datasets = {phase: ImageFolder(os.path.join(args.data_dir, phase), data_transforms[phase]) for phase in ['train', 'val']}
    dataloaders = {phase: DataLoader(image_datasets[phase], batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True) for phase in ['train', 'val']}
    dataset_sizes = {phase: len(image_datasets[phase]) for phase in ['train', 'val']}
    label_class = image_datasets['train'].classes
    num_label_class = len(label_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), 
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Flatten(), 
        nn.Linear(in_features=64 * 28 * 28, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=num_label_class)
        )

    model = model.to(device)

    def train_model(model, criterion, optimizer, num_epochs):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  
                else:
                    model.eval()   

                running_loss = 0.0
                running_corrects = 0

                for images, labels in dataloaders[phase]:
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)
        return model

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)

    model_new = train_model(model, criterion, optimizer, args.epochs)

    torch.save(model_new, os.path.join(args.model_dir, 'my_model.pth'))
    torch.save(model_new.state_dict(), os.path.join(args.model_dir, 'my_model_state.pth'))
