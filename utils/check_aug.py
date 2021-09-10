import utils.augmentation
import matplotlib.pyplot as plt
import numpy as np

class aug_config:

    rotation_augmentation = True
    brightness_augmentation = True
    shift_augmentation = True
    random_noise_augmentation = True
    rotation_angles = [12]
    pixel_shifts = [12]

def plot_sample(image, keypoint, axis, title):
  image = image.reshape(96,96)
  axis.imshow(image, cmap='gray')
  axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)
  plt.title(title)
  plt.show()

def plot_rotation_augmentation(train_images, train_keypoints, images, keypoints):
  if aug_config.rotation_augmentation:
    
    rotated_train_images, rotated_train_keypoints = utils.augmentation.rotate_augmentation(images, keypoints, aug_config.rotation_angles)
    train_images = np.concatenate((train_images, rotated_train_images))
    train_keypoints = np.concatenate((train_keypoints, rotated_train_keypoints))
    fig, ax = plt.subplots()
    plot_sample(rotated_train_images[19], rotated_train_keypoints[19], ax, "Rotation Augmentation")  


def plot_brightness_augmentation(train_images, train_keypoints, images, keypoints):
  if aug_config.brightness_augmentation:
    altered_brightness_images, altered_brightness_keypoints = utils.augmentation.alter_brightness(images, keypoints)
    train_images = np.concatenate((train_images, altered_brightness_images))
    train_keypoints = np.concatenate((train_keypoints, altered_brightness_keypoints))
    fig, axis = plt.subplots()
    plot_sample(altered_brightness_images[19], altered_brightness_keypoints[19], axis, "Alter Brightness Augmentation")