import os
import warnings
from skimage import io, morphology
import numpy as np
from tqdm import tqdm
from config import closed_post_masks, closed_pre_masks, HU_post_jpgs, HU_pre_jpgs

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

def process_images(input_folder, output_folder):

    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
        image = io.imread(image_path, as_gray=True)
        
        # Ensure the image is binary
        binary_image = (image > 0.5).astype(np.uint8)
        
        # Perform closing operation
        closed_image = morphology.binary_closing(binary_image, morphology.square(3))
        
        # Perform opening operation
        # opened_image = morphology.binary_opening(closed_image, morphology.square(1))
        
        # Save the processed image
        output_path = os.path.join(output_folder, image_file)
        io.imsave(output_path, (closed_image * 255).astype(np.uint8))

if __name__ == "__main__":
    # post_contrast
    input_folder = HU_post_jpgs
    output_folder = closed_post_masks
    os.makedirs(output_folder, exist_ok=True)
    process_images(input_folder, output_folder)

    # pre_constrast
    input_folder = HU_pre_jpgs
    output_folder = closed_pre_masks
    os.makedirs(output_folder, exist_ok=True)
    process_images(input_folder, output_folder)
