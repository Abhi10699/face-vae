import os
import torchvision.transforms as T
import glob

from PIL import Image
from torch.utils.data import Dataset, DataLoader

# class load data

DATA_PATH = "./archive"
DIR_CONTENTS = glob.glob(f"{DATA_PATH}/*.jpg", recursive=True)[:10_000]
IMAGE_SIZE = 64


# torch transofmrs

preprocess = T.Compose([
  T.Resize((IMAGE_SIZE, IMAGE_SIZE)),             
  T.ToTensor(),                     
  # T.Normalize(
  #   mean=[0.5, 0.5, 0.5],
  #   std=[0.5, 0.5, 0.5]
  # )
])

class FaceDataset(Dataset):
  def __init__(self, image_paths, transform=None):
    self.image_paths = image_paths
    self.transform = transform
  
  def __len__(self):
    return len(self.image_paths)
  
  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    image = Image.open(img_path).convert('RGB')  
    
    if self.transform:
      image = self.transform(image)
    
    return image

face_dataset = FaceDataset(image_paths=DIR_CONTENTS, transform=preprocess)
dataloader = DataLoader(face_dataset, batch_size=32, shuffle=True)