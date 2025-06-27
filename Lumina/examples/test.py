
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import numpy as np

import torch

def get_masked_image(image, fill_mask):
    w, h = image.size
    
    white_bg = Image.new('RGB', (w, h), (255, 255, 255))
    mask_pil = Image.fromarray(((1 - fill_mask).squeeze().numpy() * 255).astype(np.uint8)).convert('L')
    mask_pil.save('mask.png')
    masked_image = Image.composite(image, white_bg, mask_pil)
    return masked_image


if __name__ == "__main__":
    img_path = '33.jpg'
    img = Image.open(img_path)
    w, h = img.size
    fill_mask = torch.zeros((1, 1, h, w))
    fill_mask[:, :, 1175:, :] = 1
    
    masked_img = get_masked_image(img, fill_mask)
    masked_img.save('33_mask.png')
    # masked_img.show()
    # plt.savefig('masked_image.png', masked_img)
    print(img.size)