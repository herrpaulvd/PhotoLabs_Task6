
import numpy as np
import torch
import os
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from sklearn.metrics import accuracy_score
import random

torch.cuda.empty_cache()

def add_pad(img, shape):
    color_pick = img[0][0]
    padded_img = color_pick * np.ones(shape + img.shape[2:3], dtype=np.uint8)
    x_offset = int((padded_img.shape[0] - img.shape[0]) / 2)
    y_offset = int((padded_img.shape[1] - img.shape[1]) / 2)
    padded_img[x_offset:x_offset + img.shape[0], y_offset:y_offset + img.shape[1]] = img
    return padded_img

def resize(img, shape):
    scale = min(shape[0] * 1.0 / img.shape[0], shape[1] * 1.0 / img.shape[1])
    if scale != 1:
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return img

# Задание 2. Реализуйте класс-наследник Dataset. Он должен возвращать по индексу
class CustomDataset(Dataset):
    def __init__(self, filenames, labels):
        self._filenames = list(filenames)
        self._labels = list(labels)

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        filename = self._filenames[idx]
        label = self._labels[idx]
        img = cv2.imread(filename)
        
        shape = (224, 224)
        img = resize(img, shape)
        img = add_pad(img, shape)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1) / 255.
        return img, label

classes = [(0, 'cat'), (1, 'dog'), (2, 'coala'), (3, 'pig'), (4, 'snake')]
files = []
for idx, className in classes:
  directory = './content/' + className
  for file in os.listdir(directory):
    files.append((os.path.join(directory, file), idx))

# Задание 1. Разбейте filenames и labels на train и test части 70/30
train_filenames, test_filenames, train_labels, test_labels = [], [], [], []
# всего будет 642, возьмём только 640, 64 * 7 / 64 * 3
random.shuffle(files)

cnt_train = 64 * 7
cnt_test = 64 * 3
for i in range(cnt_train):
  f, l = files[i]
  train_filenames.append(f)
  train_labels.append(l)

for i in range(cnt_train, cnt_train + cnt_test):
  f, l = files[i]
  test_filenames.append(f)
  test_labels.append(l)

train_dataset = CustomDataset(train_filenames, train_labels)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, num_workers=0)

# Задание 3. Сделайте dataloader для test
test_dataset = CustomDataset(test_filenames, test_labels)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=8, num_workers=0)

model = resnet34(pretrained=True) # resnet обученный на ImageNet
for param in model.parameters():
  param.requires_grad=False

# loss и optimizer
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model.to('cuda')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Эта функция считает точность модели - на вход передается сама модель, номер эпохи и тестовый лоадер.
def run_test_on_epoch(model, epoch, test_loader):
    model.eval()
    with torch.no_grad():
      accuracies = []
      for batch_x, batch_y in tqdm(test_loader):
          outputs = model(batch_x.to('cuda')).detach().cpu().numpy()
          test_accuracy = []
          test_real = []
          #print(outputs)
          test_accuracy.append(outputs)
          #print(test_accuracy)
          test_real.append(batch_y.detach().cpu().numpy())
          #print(test_real)
          accuracies.append(accuracy_score(np.hstack(test_real), np.argmax(np.hstack(test_accuracy), axis=1)))
      print("Epoch", epoch, "test accuracy", np.average(np.array(accuracies)))
    model.train()

# Эта функция запихивает картинку для тренировки
def run_train_on_epoch(model, epoch, train_loader):
  for batch in train_loader:
    optimizer.zero_grad()
    image, label = batch
    image = image.to('cuda')
    label = label.to('cuda')
    label_pred = model(image)
    loss = criterion(label_pred, label)
    loss.backward()
    optimizer.step()

# Задание 5. Напишите код для обучения модели 25 эпох. В конце каждой эпохи вызывайте run_test_on_epoch() чтобы следить за точностью
for epoch in tqdm(range(25)):
  # что-то очень важное здесь
  run_train_on_epoch(model, epoch, train_dataloader)
  # половину сделал за вас
  run_test_on_epoch(model, epoch, test_dataloader)
