import cv2
from matplotlib.artist import get
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image


def get_palette_map(image, num_colors=8):
    """Convert image to blocks of specified number of colors"""
    image = image.filter(ImageFilter.GaussianBlur(12))
    w, h = image.size
    # Resize image to reduce details
    small_img = image.resize((w // 32, h // 32), Image.Resampling.NEAREST)
    
    # Convert to numpy array for processing
    img_array = np.array(small_img)
    pixels = img_array.reshape(-1, 3)
    
    # Use K-means clustering to find main colors
    pixels = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Replace each pixel with its nearest cluster center color
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(img_array.shape)
    
    # Convert back to PIL image and resize to original size
    palette_map = Image.fromarray(quantized)
    palette_map = palette_map.resize((w, h), Image.Resampling.NEAREST)  # Use NEAREST to keep block boundaries clear
    
    palette_map_tensor = torch.tensor(np.array(palette_map)/255.0).permute(2, 0, 1)
    save_image(palette_map_tensor, "./examples/22_palette_2.jpg")
    return palette_map


if __name__ == "__main__":
   image = Image.open("./examples/22.jpg")
   palette_map = get_palette_map(image, 8)
   palette_map.save("./examples/22_palette.jpg")