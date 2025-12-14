import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, Resize, Compose, Pad
import torchvision.utils as utils

from model import Generator

def test_model(model_path, test_image_path, output_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(color_mode="L").to(device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    
    test_image = Image.open(test_image_path).convert('L')
    original_size = test_image.size

    
    pad_height = (32 - (original_size[0] % 32)) % 32
    pad_width = (32 - (original_size[1] % 32)) % 32
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    print(pad_top, pad_bottom, pad_left, pad_right)

    
    transform = Compose([
        Pad((pad_top, pad_bottom, pad_left, pad_right)),
        ToTensor()
    ])

    print(f"Original size: {original_size}")
    test_image = transform(test_image).unsqueeze(0).to(device)
    print(f"Padded size: {test_image.shape}")
    print(test_image.shape)
    
    with torch.no_grad():
        output_image = generator(test_image)

    
    utils.save_image(output_image, output_path)


if __name__ == "__main__":
    model_path = "checkpoints/netG_epoch_10.pth"  # Path to the trained generator model
    for i in range(19):
        test_image_path = f"test_images/{i}.png"  # Path to the test image
        output_path = f"results/{i}.png"  # Path to save the reconstructed image
        test_model(model_path, test_image_path, output_path)

    
    os.makedirs("results", exist_ok=True)

    
    #test_model(model_path, test_image_path, output_path)
