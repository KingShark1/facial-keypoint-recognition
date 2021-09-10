import cv2
from math import cos, sin, pi
import numpy as np

def rotate_augmentation(images, keypoints, rotation_angles):
  """
  """
  rotated_images = []
  rotated_keypoints = []
  for angle in rotation_angles:
    for angle in [angle, -angle]:
      M = cv2.getRotationMatrix2D((48,48), angle, 1.)
      angle_rad = -angle*pi/180.
      for image in images:
        rotated_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
        rotated_images.append(rotated_image)
      for keypoint in keypoints:
        rotated_keypoint = keypoint - 48.
        for idx in range(0, len(rotated_keypoint), 2):
          rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)
          rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)
        rotated_keypoint += 48.   
        rotated_keypoints.append(rotated_keypoint)
          
  return np.reshape(rotated_images,(-1,96,96,1)), rotated_keypoints

def alter_brightness(images, keypoints):
  """
  """
  altered_brightness_images = []
  
  inc_brightness_images = np.clip(images*1.2, 0.0, 1.0)    
  dec_brightness_images = np.clip(images*0.6, 0.0, 1.0)    
  
  altered_brightness_images.extend(inc_brightness_images)
  altered_brightness_images.extend(dec_brightness_images)
  
  return altered_brightness_images, np.concatenate((keypoints, keypoints))
