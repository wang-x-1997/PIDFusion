import os
import numpy as np
import torch.utils.data as data1
import torchvision.transforms as transforms
# from scipy.misc import imread, imresize  #1.2.1
import torch
from PIL import Image
import cv2

def load_image(x):
  imgSB = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
  # imgB = np.swapaxes(imgB, 0, 2)
  # imgB = np.swapaxes(imgB, 1, 2)
  imgSB = imgSB.astype('float32')
  imgSB = torch.from_numpy(imgSB)
  # imgSB = imgSB.unsqueeze(0)
  imgSB = imgSB.unsqueeze(0)
  return imgSB

def make_dataset(root, train=True):
    dataset = []

    if train:
      dir_img = os.path.join(r'D:\Image_Data\RoadScene-master\256')

    for index in range(320):

      imgA =  str(index+1) + '_1.jpg'
      imgB = str(index+1) + '_2.jpg'
      # imgC = 'A'+str(index+1) + '.jpg'
      # imgA =  str(index+1) + '-2.jpg'
      # imgB =  str(index+1) + '-1.jpg'
      # imgC =  'A'+str(index+1) + '.jpg'
      dataset.append([os.path.join(dir_img, imgA), os.path.join(dir_img, imgB)])


    return dataset


class fusiondata(data1.Dataset):

  def __init__(self, root, transform=None, train=True):
    self.train = train

    if self.train:
      self.train_set_path = make_dataset(root, train)

  def __getitem__(self, idx):
    if self.train:
      imgA_path, imgB_path = self.train_set_path[idx]

      imgA = load_image(imgA_path)
      imgB = load_image(imgB_path)

      return imgA,imgB

      
  def __len__(self):
    if self.train:
      return 320 #2792

