from models.AdaIN import StyleTransfer
from glob import glob
import random
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)

# Initialize the style transfer model
StyleTransfermodel = StyleTransfer()
folders = glob("/data/temp_zenglin/data/imagenet/train/*")

# Randomly select 20 folders
selected_folders = random.sample(folders, 20)

# Define a transform to resize and convert images to tensor
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),  # Convert to RGB (three channels)
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def save_concat_images(tensors, output_path):
    # Convert tensors to numpy arrays
    # images = [tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor for tensor in tensors]
    #
    # # Ensure the images are in the format (H, W, C)
    # images = [img.transpose((1, 2, 0)) if img.shape[0] in [3, 4] else img for img in images]
    # images = [img[:, :, [2, 1, 0]] for img in images]  # Convert BGR to RGB
    images = []
    for tensor in tensors:
        if torch.is_tensor(tensor):
            # Normalize the tensor to 0-1
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            # Convert tensor to numpy array
            img = tensor.detach().cpu().numpy()
            # Transpose from C, H, W to H, W, C
            img = img.transpose((1, 2, 0))
            # Convert BGR to RGB
            img = img[:, :, [2, 1, 0]]
            images.append(img)
    # Get dimensions for the new image
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    total_width = sum(widths)
    max_height = max(heights)

    # Create an empty array for the new concatenated image/
    new_image_array = np.zeros((max_height, total_width, 3), dtype=images[0].dtype)

    # Paste images side by side into the new image array
    x_offset = 0
    for img in images:
        height = img.shape[0]
        width = img.shape[1]
        new_image_array[:height, x_offset:x_offset + width] = img
        x_offset += width

    # Display and save the concatenated image using imshow and savefig
    plt.imshow(new_image_array)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.clf()


# Processing each folder
for i, folder in tqdm(enumerate(selected_folders)):
    images = glob(folder + "/*.JPEG")
    selected_images = random.sample(images, 16)  # Select up to 16 images

    # Load images as tensors
    tensors = [transform(Image.open(img)).to("cuda") for img in selected_images]

    # For each image, pair with all images from the other folders
    for j, other_folder in enumerate(selected_folders):
        if i != j:  # Ensure not the same folder
            other_images = glob(other_folder + "/*.JPEG")
            other_selected_images = random.sample(other_images, min(16, len(other_images)))
            other_tensors = [transform(Image.open(img)).to("cuda") for img in other_selected_images]

            for tensor1, img1 in zip(tensors, selected_images):
                for tensor2, img2 in zip(other_tensors, other_selected_images):
                    # Style transfer
                    combined_tensor = StyleTransfermodel(tensor1.unsqueeze(0), tensor2.unsqueeze(0), alpha=1)

                    # Convert tensors to PIL Images
                    content_image = tensor1
                    style_image = tensor2
                    combined_image = combined_tensor.squeeze(0)

                    # Save concatenated images
                    output_folder = os.path.join("./style_sample/",
                                                 f"{os.path.basename(folder)}_{os.path.basename(other_folder)}")
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, f"{os.path.basename(img1)}_{os.path.basename(img2)}.jpg")
                    save_concat_images([content_image, style_image, combined_image], output_path)
