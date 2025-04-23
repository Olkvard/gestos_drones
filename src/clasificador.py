import numpy as np
from PIL import Image

import torch
import torch.optim as optim
import torch.utils as utils
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

import os
import glob
import time
import json

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
USE_PRETRAINED_MODEL = False
USE_AUGMENTATION = True
NUM_EPOCHS = 1000
data_path = "data"
print("\nImportando datos...")

def preprocess(USE_AUGMENTATION = False):
    layers = []
    layers.append(v2.ToImage())
    layers.append(v2.ToDtype(torch.uint8, scale=True))
    layers.append(v2.Resize(size=(128, 128), antialias=True))

    if USE_AUGMENTATION:
        layers.extend((
            v2.RandomApply([v2.RandomAffine(degrees=(-45, 45))], p=0.5),
            v2.RandomApply([v2.RandomAffine(degrees=0, translate=(0.2, 0.2))], p=0.5),
            v2.RandomApply([v2.RandomAffine(degrees=0, scale=(0.8, 1.2))], p=0.5),
            v2.RandomApply([v2.RandomAffine(degrees=0, shear=(-10, 10))], p=0.5),
        ))
    layers.append(v2.ToDtype(torch.float32, scale=True))
    layers.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return v2.Compose(layers)

class Dataset(utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = torch.tensor(self.labels[index], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label  
    
def train_batch(model, batch_x, batch_y, criterion, optimizer):
    logits = model(batch_x)
    optimizer.zero_grad()
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()

def evaluate_batch(device, model, batch_x, batch_y, criterion, top_k):
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    m = len(batch_x)
    logits = model(batch_x)
    loss = criterion(logits, batch_y).item()
    accuracy_top1 = (torch.max(logits.data, 1)[1] == batch_y).sum().item()
    k_labels = torch.topk(input=logits.data, k=top_k, dim=1, largest=True, sorted=True)[1]
    accuracy_topk = (~torch.prod(input=torch.abs(batch_y.unsqueeze(1) - k_labels), dim=1).to(torch.bool)).long().sum().item()
    loss /= m
    accuracy_top1 /= m
    accuracy_topk /= m
    return loss, accuracy_top1, accuracy_topk

def train_epoch(device, model, train_dataloader, criterion, optimizer, eval_batch_step=1, top_k=5):
    end_batch_time=time.time()
    for step, (batch_x, batch_y) in enumerate(train_dataloader):
        start = time.time()
        loading_duration = start-end_batch_time
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        model.train()
        train_batch(model, batch_x, batch_y, criterion, optimizer)
        end_training_time = time.time()
        training_duration = end_training_time - start
        if eval_batch_step > 0 and (step == 0 or (step + 1) % eval_batch_step == 0):
            model.eval()
            with torch.no_grad():
                loss, accuracy_top1, accuracy_topk = evaluate_batch(device, model, batch_x, batch_y, criterion, top_k=top_k)
            end_batch_time = time.time()
            eval_duration = end_batch_time - end_training_time
            batch_duration = loading_duration + training_duration + eval_duration
            print(
                "\t[batch {0}/{1}] total={3:.3f}s | load_time={4:.3f}s, train_time={5:.3f}s, eval_time={6:.3f}s, loss={7:.4f}, acc1={8:.4f}, acc{2}={9:.4f}".format(
                    step + 1, len(train_dataloader), top_k, batch_duration, loading_duration, training_duration, eval_duration, loss, accuracy_top1, accuracy_topk)
            )
        end_batch_time=time.time()

def evaluate_epoch(device, model, dataloader, criterion, top_k):
    n = len(dataloader.sampler)
    accuracy_top1 = 0.0
    accuracy_topk = 0.0
    loss = 0.0
    for batch_x, batch_y in dataloader:
        m = len(batch_x)
        l, acc1, acck = evaluate_batch(device, model, batch_x, batch_y, criterion, top_k=top_k)
        loss += m * l
        accuracy_top1 += m * acc1
        accuracy_topk += m * acck
    loss /= n
    accuracy_top1 /= n
    accuracy_topk /= n
    return loss, accuracy_top1, accuracy_topk

def train_model(device, model, optimizer, lr_scheduler, criterion, train_dataloader, val_dataloader, best_loss, training_metrics, validation_metrics, num_epochs= 100, eval_batch_step=0, eval_epoch_step=1, top_k=5):
    model.train()
    last_epoch = lr_scheduler.last_epoch
    print("\nEntrenando modelo [last_epoch: {0}, num_epochs: {1}, batch_size: {2}]...".format(last_epoch, num_epochs, train_dataloader.batch_size))

    for epoch in range(last_epoch, num_epochs + last_epoch):
        start = time.time()
        train_epoch(device=device, model=model, train_dataloader=train_dataloader, optimizer=optimizer, criterion=criterion, eval_batch_step=eval_batch_step, top_k=top_k)
        lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()
        end_training_time = time.time()
        training_duration = end_training_time - start
        if eval_epoch_step > 0 and (epoch == 0 or (epoch + 1) % eval_epoch_step == 0):
            model.eval()
            with torch.no_grad():
                training_loss, training_accuracy_top1, training_accuracy_topk = evaluate_epoch(device=device, model=model, dataloader=train_dataloader, criterion=criterion, top_k=top_k)
                end_evaluation_time = time.time()
                evaluation_duration = end_evaluation_time - end_training_time
                validation_loss, validation_accuracy_top1, validation_accuracy_topk = evaluate_epoch(device=device, model=model, dataloader=val_dataloader, criterion=criterion, top_k=top_k)
            end_epoch_time = time.time()
            validation_duration = end_epoch_time - end_evaluation_time
            epoch_duration = end_epoch_time - start
            print(
                "[epoch {0}/{1}] [lr = {2:.8f}\ttotal = {4:.3f}s | train : train_time = {5:.3f}s, eval_time = {6:.3f}s, loss = {7:.4f}, acc1 = {8:.4f}, acc{3} = {9:.4f} | val : eval_time = {10:.3f} loss = {11:.4f}, acc1 = {12:.4f}, acc{3} = {13:.4f}]".format(
                    epoch + 1, last_epoch + num_epochs, lr, top_k, epoch_duration, training_duration, evaluation_duration,
                    training_loss, training_accuracy_top1, training_accuracy_topk,
                    validation_duration, validation_loss, validation_accuracy_top1, validation_accuracy_topk)
            )
            training_metrics.append((training_loss, training_accuracy_top1, training_accuracy_topk))
            validation_metrics.append((validation_loss, validation_accuracy_top1, validation_accuracy_topk))
        else:
            training_metrics.append((None, None, None))
            validation_metrics.append((None, None, None))
        if validation_loss < best_loss:
            best_loss = validation_loss
            checkpoint = {
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                'training_metrics' : training_metrics,
                'validation_metrics' : validation_metrics,
                'best_loss' : best_loss,
                'epoch' : epoch
            }
            torch.save(checkpoint, 'checkpoint.pth')
            print("Modelo guardado en checkpoint.pth")
        print("Tiempo total de entrenamiento: {0:.3f}s".format(time.time() - start))


def predict(model, imgs, top_k=5):
    imgs = imgs.to(DEVICE)
    model.eval()
    with torch.no_grad():
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        logits = nn.Softmax(dim=1)(model(imgs))
        preds = torch.topk(input=logits, k=top_k, dim=1, largest=True, sorted=True)
        keys = preds[1].squeeze().tolist()
        values = preds[0].squeeze().tolist()
    return keys, values

def main():
    gesture_folders = sorted([folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))])
    sorted_labels = np.asarray(gesture_folders)
    n_classes = len(sorted_labels)

    print("Clases detectadas: ", sorted_labels)

    images, labels = [], []

    for folder in gesture_folders:
        folder_path = os.path.join(data_path, folder)
        for file in glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.jpg")):
            images.append(os.path.abspath(file))
            label = sorted_labels.tolist().index(folder)
            labels.append(label)

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    images_dict = {'train': train_images, 'val': val_images}
    labels_dict = {'train': train_labels, 'val': val_labels}

    print("\nPreprocesando datos...")

    train_dataset = Dataset(images_dict['train'], labels_dict['train'], transform=preprocess(USE_AUGMENTATION))
    val_dataset = Dataset(images_dict['val'], labels_dict['val'], transform=preprocess(False))

    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=512, num_workers=2, shuffle=True)
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=512, num_workers=2, shuffle=False)

    print("\nInicializando modelo...")
    model = models.efficientnet_b0(weights=None, num_classes=n_classes)
    model.to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
    lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1)
    criterion = nn.CrossEntropyLoss(weight=None, reduction='sum')
    best_loss = np.inf
    if os.path.exists("models/best_model"):
        best_loss = torch.load('models/best_model', map_location=torch.device(DEVICE))['validation_metrics'][-1][0]

    if USE_PRETRAINED_MODEL and os.path.exists("models/best_model"):
        print("\nCargando modelo antiguo...")
        checkpoint = torch.load('models/best_model', map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        training_metrics = checkpoint['training_metrics']
        validation_metrics = checkpoint['validation_metrics']
    else:
        training_metrics, validation_metrics = [], []

    if NUM_EPOCHS > 0: 
        train_model(
            device=DEVICE,
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            criterion=criterion,
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
            best_loss=best_loss, 
            training_metrics=training_metrics, 
            validation_metrics=validation_metrics,
            num_epochs=NUM_EPOCHS,
            eval_batch_step=0, 
            eval_epoch_step=1,
            top_k=2
        )
    with Image.open('banana.jpg') as img:
        img = preprocess(False)(img.convert('RGB'))
        keys, values = predict(model, img, top_k=2)
        dictionary = dict(zip(sorted_labels[keys], values))
        print(json.dumps(dictionary, indent=4))

if __name__ == "__main__":
    main()
