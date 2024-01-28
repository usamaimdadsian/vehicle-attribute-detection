# %%
import os, torch
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset,DataLoader,random_split
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim


# %%
FOLDER_DATASET = "/home/kk/Desktop/usama/datasets/VeRi"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 10

# %%
print(DEVICE)

# %% [markdown]
# # Implementations

# %%
def test_accuracy(model, test_loader):
  with torch.no_grad():
      total_correct = 0
      total_samples = 0
      for images, labels in test_loader:
          images, labels = images.to(DEVICE), labels.to(DEVICE)
          outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          total_correct += (predicted == labels).sum().item()
          total_samples += labels.size(0)

      accuracy = total_correct / total_samples
      print(f'Test Accuracy On best Model: {accuracy:.4f}')

# %%
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




# %%
import copy
def train(epochs, model, criterion, optimizer, train_loader, valid_loader, test_loader):
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
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          _, predicted = torch.max(outputs, 1)
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
              outputs = model(images)
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
          outputs = best_model(images)
          _, predicted = torch.max(outputs, 1)
          total_correct += (predicted == labels).sum().item()
          total_samples += labels.size(0)

      accuracy = total_correct / total_samples
      print(f'Test Accuracy On best Model: {accuracy:.4f}')
  return best_model

# %% [markdown]
# # Main

# %%
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = CustomDataset(FOLDER_DATASET+"/image_train",FOLDER_DATASET+"/train_label.xml",FOLDER_DATASET+"/list_color.txt",transform)

train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset,[train_size,val_size])



test_dataset = CustomDataset(FOLDER_DATASET+"/image_test",FOLDER_DATASET+"/test_label.xml",FOLDER_DATASET+"/list_color.txt",transform)

# %%

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


# %%
# Initialize the model, loss function, and optimizer
# model = VehicleColorRecognitionModel().to(DEVICE)

model = models.resnet18(pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


best_model = train(EPOCHS,model, criterion, optimizer, train_loader, valid_loader, test_loader)

torch.save(best_model,"veri_resnet18.pt")

