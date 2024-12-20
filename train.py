import os
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np


setting = {
'num_epochs' : 75,
'sigma_value' : 10,
'batch_size' : 256,
'decay'      : True
}






num_epochs = setting['num_epochs']
sigma_value = setting['sigma_value']
decay_flag = setting['decay']
batch_size = setting['batch_size']




def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs.squeeze(), labels[:, 0])  # Labels only contain age
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

#Evaluation on Test Set using MSE Score
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim= 1)
            true_labels.append(labels[:, 0].cpu().numpy())  # Store the true age labels
            predicted_labels.append(outputs.cpu().squeeze().numpy())  # Store the predicted age labels


    # Convert list of arrays to single arrays
    true_labels = np.concatenate(true_labels)
    predicted_labels = np.concatenate(predicted_labels)

    # Calculate R^2 score for age
    r2_age = mean_squared_error(true_labels, predicted_labels)

    print(f"R^2 Score for Age: {r2_age:.4f}")




class FaceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., image preprocessing).
        """
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        
        # Load image
        image = Image.open(img_name)
        
        # Parse the filename to get the labels
        filename = os.path.basename(img_name).split('.')[0]
        try:
            age, gender, race, _ = filename.split('_')
        except:
            age, gender, race, _ = 0, 0, 0, 0
            
        # Convert labels to appropriate types
        age = int(age)
        gender = int(gender)
        race = int(race)

        # Apply transformation to the image if specified
        if self.transform:
            image = self.transform(image)

        # Return image and the labels as a tuple
        return image, torch.tensor([age, gender, race], dtype=torch.float)

#Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert image to tensor
])

# Specify the directory containing the images
img_dir = 'crop_part1'

# Initialize the dataset
face_dataset = FaceDataset(img_dir=img_dir, transform=transform)


class AgePredictorCNN(nn.Module):
    def __init__(self):
        super(AgePredictorCNN, self).__init__()        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Adjust based on input image size
        self.fc2 = nn.Linear(256, 117)  # Output for age regression        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)        
    def forward(self, x):
        # Convolutional layers with ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv3(x)))  # 16x16 -> 8x8        
        # Flatten the output
        x = x.view(-1, 128 * 16 * 16)  # Adjust based on input image size       
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)       
        # Age regression output
        x = self.fc2(x)
        
        return x
    

def create_gaussian_mound(num_classes, correct_class, sigma=1):
    x = np.arange(num_classes)
    target_probs = np.exp(-0.5 * ((x - correct_class) / sigma) ** 2)
    target_probs /= np.sum(target_probs)  # Normalize to sum to 1
    return torch.tensor(target_probs)

def kl_divergence(p, q):
    p = p + 1e-8  # Adding small epsilon to avoid log(0)
    return torch.sum(p * torch.log(p / q))




# Example of using a DataLoader
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
# Example usage:
model = AgePredictorCNN()
model.to(device)
# Example of setting up the optimizer and loss function for training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

dataloader = DataLoader(face_dataset, batch_size=256, shuffle=True)


train_size = int(0.8 * len(face_dataset))
test_size = len(face_dataset) - train_size
train_dataset, test_dataset = random_split(face_dataset, [train_size, test_size])

# 2. Create DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)




for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    batch = 1
    for images, labels in train_loader:
        # Move tensors to the appropriate device (e.g., GPU if available)
        images, labels = images.to(device), labels.to(device)
        labels = labels[:,0].long()
        
        # Forward pass
        logits = model(images)
        predicted_probs = torch.softmax(logits, dim=1)
        # Loss calculation (CrossEntropyLoss expects raw logits, not softmax)
        loss = criterion(logits, labels)
        true_loss = loss
        
        
        
        kl_loss = 0
        for i, label in enumerate(labels):
            smooth_target = create_gaussian_mound(117, label.cpu().numpy(), sigma= sigma_value)
            smooth_target = smooth_target.to(device)
            kl_loss += kl_divergence(smooth_target, predicted_probs[i])
        
        kl_loss /= len(labels)
        
        loss = loss + kl_loss
        # Backward pass and optimization
        loss.backward()
        if batch % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()
        # Accumulate loss
        running_loss += true_loss.item()
        batch += 1
    evaluate_model(model, train_loader)
    
    if decay_flag:
        sigma_value = sigma_value*.95



    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/batch}')

print('Training complete.')


