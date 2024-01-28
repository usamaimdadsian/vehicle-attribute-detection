
import torch
import torchvision
import numpy as  np
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
from PIL import Image
import os
from torch.utils.data import Dataset,DataLoader,random_split
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self,root_dir, xml_file,color_file, transform=None ) -> None:
        self.root_dir = root_dir
        self.color_mapping = self.load_color_mapping(color_file)
        self.data = self.parse_xml(xml_file)
        self.transform = transform

    def load_color_mapping(self,color_file):
        color_mapping = {}
        with open(color_file,'r') as file:
            for line in file:
                color_id, color_name = line.strip().split(' ',1)
                color_mapping[int(color_id)-1] = color_name
        return color_mapping

    def parse_xml(self, xml_file):
        data = []
        with open(xml_file,'r') as file:
            tree = ET.fromstring(file.read())
            tree = ET.ElementTree(tree)
            root = tree.getroot()
            for item in root.findall('.//Item'):
                image_name = item.get('imageName')
                color_id = int(item.get('colorID'))
                data.append((image_name,color_id))
        return data

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_name, color_id = self.data[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, color_id-1



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder_output_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def encoder_output_size(self):
        encoded = self.encoder(torch.empty(1,3,SIZE,SIZE))
        return encoded.view(encoded.size(0), -1).size()[-1]
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded.view(encoded.size(0), -1))
        return decoded, classification


import copy
def train(epochs, model, criterion_autoencoder , criterion_classifier, optimizer, train_loader, valid_loader, test_loader):
  best_model = None
  best_acc = 0

  # Training loop
  for epoch in range(epochs):
      model.train()
      total_train_correct = 0
      total_train_samples = 0
      for images, labels in train_loader:
          images, labels = images.to(DEVICE), labels.to(DEVICE)
          optimizer.zero_grad()

          outputs_autoencoder, outputs_classifier = model(images)

          loss_autoencoder = criterion_autoencoder(outputs_autoencoder, images)
          loss_classifier = criterion_classifier(outputs_classifier, labels)

          total_loss = loss_autoencoder + 0.1 * loss_classifier

          total_loss.backward()
          optimizer.step()

          _, predicted = torch.max(outputs_classifier, 1)
          total_train_correct += (predicted == labels).sum().item()
          total_train_samples += labels.size(0)
      train_accuracy = total_train_correct / total_train_samples

      # Validation loop
      model.eval()
      with torch.no_grad():
          total_correct = 0
          total_samples = 0
          for images, labels in valid_loader:
              images, labels = images.to(DEVICE), labels.to(DEVICE)
              _,outputs = model(images)
              _, predicted = torch.max(outputs, 1)
              total_correct += (predicted == labels).sum().item()
              total_samples += labels.size(0)

          accuracy = total_correct / total_samples
          print(f'Epoch [{epoch+1}/{epochs}], Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {accuracy:.4f}')

          if (accuracy > best_acc):
            print("New Best Model with Accuracy: ", accuracy)
            best_acc = accuracy
            best_model = copy.deepcopy(model)

  print("Training finished.")

  # Testing the model
  model.eval()
  with torch.no_grad():
      total_correct = 0
      total_samples = 0
      for images, labels in test_loader:
          images, labels = images.to(DEVICE), labels.to(DEVICE)
          _,outputs = best_model(images)
          _, predicted = torch.max(outputs, 1)
          total_correct += (predicted == labels).sum().item()
          total_samples += labels.size(0)

      accuracy = total_correct / total_samples
      print(f'Test Accuracy On best Model: {accuracy:.4f}')
  return best_model

def test_accuracy(model, test_loader):
  with torch.no_grad():
      total_correct = 0
      total_samples = 0
      for images, labels in test_loader:
          images, labels = images.to(DEVICE), labels.to(DEVICE)
          _,outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          total_correct += (predicted == labels).sum().item()
          total_samples += labels.size(0)

      accuracy = total_correct / total_samples
      print(f'Test Accuracy On best Model: {accuracy:.4f}')


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = CustomDataset("VeRi/image_train","VeRi/train_label.xml","VeRi/list_color.txt",transform)

    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset,[train_size,val_size])



    test_dataset = CustomDataset("VeRi/image_test","VeRi/test_label.xml","VeRi/list_color.txt",transform)

    BATCH_SIZE = 32
    NUM_CLASSES = 10
    EPOCHS = 10
    SIZE = 224

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # model = Autoencoder().to(DEVICE)
    # classifier_criterion = nn.CrossEntropyLoss()
    # autoencoder_criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)


    # best_model = train(EPOCHS,model,autoencoder_criterion , classifier_criterion, optimizer, train_loader, valid_loader, test_loader)



    # torch.save(best_model,"vehicle_color_classification.pt")
    # https://drive.google.com/file/d/19lTPTuu1AjcDovN0Mhujc5tloJ1stLPz/view?usp=sharing
    model = torch.load("vehicle_color_classification.pth")
    test_accuracy(model,test_loader)


