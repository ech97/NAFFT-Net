import math
import numpy as np
import torch

# 최대 신호 대 잡음비
def psnr(ori_img, con_img):
  """
  @params ori_img: 원본 이미지
  @params con_img: 비교 대상 이미지
  """
  
  # 해당 이미지의 최대값 (채널 최대값 - 최솟값)
  max_pixel = 1. if ori_img.max() <= 1 else 255.

  # MSE 계산
  mse = torch.mean((ori_img - con_img)**2)

  if mse == 0:
    return 100
  
  # PSNR 계산
  psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
  
  return psnr