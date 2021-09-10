import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils.loaders
import utils.augmentation
import utils.check_aug



def main():

  train_file = pd.read_csv('/home/kingshark1/Competitions/facial-keypoint-recognition/data/training.csv')
  test_file = pd.read_csv('/home/kingshark1/Competitions/facial-keypoint-recognition/data/test.csv')

  clean_train_file = train_file.dropna()
  train_file = train_file.fillna(method='ffill')
  
  train_images = utils.loaders.load_images(train_file)
  images = utils.loaders.load_images(clean_train_file)
  train_keypoints = utils.loaders.load_keypoints(train_file)
  keypoints = utils.loaders.load_keypoints(clean_train_file)
  test_images = utils.loaders.load_images(test_file)
  
  print("Loaded Files sucessfully")

  # For checking rotation 
  utils.check_aug.plot_rotation_augmentation(train_images, train_keypoints, images, keypoints)

  # For checking Brigtness augmentation
  utils.check_aug.plot_brightness_augmentation(train_images, train_keypoints, images, keypoints)

  print('Exiting main function')

if __name__ == '__main__':
  main()
  print('Out of main function')