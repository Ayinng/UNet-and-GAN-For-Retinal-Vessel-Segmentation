import cv2
import numpy as np


def calculate_psnr(img1_path, img2_path):
    # 读取两张图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 确保两张图片尺寸相同
    if img1.shape != img2.shape:
        raise ValueError("The two images must have the same dimensions.")

    # 计算MSE (Mean Squared Error)
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        # 当两张图片完全相同时，PSNR 值为无穷大
        return float('inf')

    # 计算 PSNR 值
    max_pixel_value = 255.0  # 对于8位图像，最大像素值为255
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr

#初始化总值
all=0
# 测试代码
for i in range (19):
    img1_path = f"Groudtruth/{i}.png"
    img2_path = f"results/{i}.png"
    psnr_value = calculate_psnr(img1_path, img2_path)
    print(f"PSNR between the {i} images: {psnr_value} dB")

print(f"PSNR in average ={all/20}")
