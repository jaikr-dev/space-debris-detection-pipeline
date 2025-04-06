# %%
"""
Space Debris Detection Pipeline

This module implements a complete pipeline for space debris detection using
computer vision and deep learning techniques. It handles data generation,
preprocessing, augmentation, model training, and evaluation.
"""

#------------------------------------------------------------------------------
# Standard library imports
#------------------------------------------------------------------------------
import os                      # Operating system interface
import re                      # Regular expressions
import time                    # Time access and conversions
import glob                    # Unix style pathname pattern expansion
import random                  # Generate random numbers
import shutil                  # High-level file operations
import subprocess              # Subprocess management
from datetime import datetime  # Date and time handling
from pathlib import Path       # Object-oriented filesystem paths
from typing import Dict, List, Tuple, Optional, Any  # Type hinting support

#------------------------------------------------------------------------------
# Data manipulation and visualization
#------------------------------------------------------------------------------
import numpy as np                   # Numerical computing library
import pandas as pd                  # Data manipulation and analysis
import matplotlib.pyplot as plt      # Plotting library
import matplotlib as mpl             # Matplotlib configuration
import matplotlib.font_manager as fm # Font management
import cv2                           # OpenCV for computer vision operations
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting functionality
import seaborn as sns                # Statistical data visualization
from matplotlib.colors import LinearSegmentedColormap  # Custom colormaps
from sklearn.metrics import (        # Metrics for model evaluation
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report
)

#------------------------------------------------------------------------------
# Deep learning and model related imports
#------------------------------------------------------------------------------
import torch                    # PyTorch deep learning framework
from ultralytics import YOLO    # YOLO object detection model
import wandb                    # Weights & Biases for experiment tracking
import albumentations as A      # Image augmentation library

# %%
#------------------------------------------------------------------------------
# Base directory configurations
#------------------------------------------------------------------------------
PROJECT_BASE = r"C:\Users\Jai\Downloads\Individual Project Folder"
BLENDER_BASE = os.path.join(PROJECT_BASE, "Blender_Renders")
YOLOV8_BASE = os.path.join(PROJECT_BASE, "yolov8-project")

#------------------------------------------------------------------------------
# Dataset directory structure
#------------------------------------------------------------------------------
DATASET_BASE = os.path.join(YOLOV8_BASE, "dataset6")
DEBRIS_DIR = os.path.join(DATASET_BASE, "Debris")
DEBRIS_SCALED_DIR = os.path.join(DATASET_BASE, "Debris_Scaled")
DEBRIS_ANNOTATED_DIR = os.path.join(DATASET_BASE, "Debris_Annotated")

#------------------------------------------------------------------------------
# Configuration file paths
#------------------------------------------------------------------------------
CONFIG_PATH = os.path.join(DEBRIS_SCALED_DIR, "config_scaled.yaml")

#------------------------------------------------------------------------------
# Blender application and script paths
#------------------------------------------------------------------------------
BLENDER_EXECUTABLE = r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
BLEND_FILE = os.path.join(BLENDER_BASE, "Earth_render", "Earth.blend")
BLENDER_SCRIPT = os.path.join(BLENDER_BASE, "Earth_render", "Crash Solution V3.py")
FLAG_FILE = os.path.join(BLENDER_BASE, "Earth_render", "stop_flag.txt")

#------------------------------------------------------------------------------
# Blender object coordinate paths
#------------------------------------------------------------------------------
ORBIT_CONTAINER_COORDINATES = os.path.join(DATASET_BASE, 'orbit_container_coordinates')
DEBRIS_TRACKING_CONTAINER_COORDINATES = os.path.join(DATASET_BASE, 'debris_tracking_container_coordinates')
CAMERA_CONTAINER_COORDINATES = os.path.join(DATASET_BASE, 'camera_container_coordinates')
QUATERNION_COORDINATES = os.path.join(DATASET_BASE, 'debris_orientation_quaternion')
EULER_COORDINATES = os.path.join(DATASET_BASE, 'debris_orientation_euler')
AUGMENTATION_SUMMARY = os.path.join(DEBRIS_SCALED_DIR, 'Augmentations Summary.txt')

#------------------------------------------------------------------------------
# Python scripts location
#------------------------------------------------------------------------------
SCRIPTS_DIR = os.path.join(BLENDER_BASE, "Python Scripts")

#------------------------------------------------------------------------------
# YOLOv8 training configurations
#------------------------------------------------------------------------------
WANDB_API_KEY = os.getenv('WANDB_API_KEY', "3a215434bfc6659b2f1ae767e669c8dcf1964a84")
WANDB_PROJECT_NAME = "yolov8-debris-detection"
MODEL_TYPE = "yolov8n.pt"
RUN_NUMBER = "Run 9"

#------------------------------------------------------------------------------
# YOLOv-n confusion matrix CSV directories. Directories must be set manually each time. 
#------------------------------------------------------------------------------
CONFUSION_MATRIX_8 = os.path.join(DEBRIS_SCALED_DIR,r'Run 9 yolov8n (Mar-25-2025 , 09-49-56)\confusion_matrix_raw.csv')
CONFUSION_MATRIX_11 = os.path.join(DEBRIS_SCALED_DIR,r'Run 10 yolo11n (Mar-27-2025 , 22-49-30)\confusion_matrix_raw.csv')
CONFUSION_MATRIX_12 = os.path.join(DEBRIS_SCALED_DIR,r'Run 11 yolo12n (Mar-28-2025 , 11-45-29)\confusion_matrix_raw.csv')

#------------------------------------------------------------------------------
# Directory creation
#------------------------------------------------------------------------------
os.makedirs(DEBRIS_SCALED_DIR, exist_ok=True)
os.makedirs(DEBRIS_ANNOTATED_DIR, exist_ok=True)

#------------------------------------------------------------------------------
# Dataset configuration
#------------------------------------------------------------------------------
SPLITS = ["train", "val", "test"]

#------------------------------------------------------------------------------
# Image dimensions
#------------------------------------------------------------------------------
ORIGINAL_SIZE = 1080
NEW_SIZE = 640

# %%
#------------------------------------------------------------------------------
# Blender Process Management Function
#------------------------------------------------------------------------------
# Check if the BLENDER_SCRIPT directory has existing .JSON and .txt files. If so, delete them before running this script.

def restart_blender(delay=10):
    """
    Continuously runs Blender in background mode to render specified .blend file and execute Python script.
    
    Args:
        delay (int): Time to wait (in seconds) before restarting Blender after closure
    
    This function is useful for batch rendering or automating rendering tasks with periodic restarts.
    It will continue until a stop flag file is created.
    """
    print("Starting Blender monitoring script...")
    
    while True:
        print("Starting Blender render process...")
        
        #----------------------------------------------------------------------
        # Launch Blender process with configuration
        #----------------------------------------------------------------------
        # Launch Blender in background mode (-b) with specified .blend file and execute Python script
        result = subprocess.run(
            [BLENDER_EXECUTABLE, "-b", BLEND_FILE, "--python", BLENDER_SCRIPT]
        )
        
        # Print the return code (0 indicates success, non-zero indicates an error)
        print(f"Blender exited with return code {result.returncode}.")

        #----------------------------------------------------------------------
        # Process control and monitoring
        #----------------------------------------------------------------------
        # Check if the flag file exists, indicating batch number limit reached
        if os.path.exists(FLAG_FILE):
            print("Batch number limit reached. Stopping the monitoring script.")
            break
        
        # Wait for specified delay before restarting Blender
        print(f"Waiting {delay} seconds before restarting...")
        time.sleep(delay)
        
        print("Restarting Blender...")

#------------------------------------------------------------------------------
# Function execution (disabled by default)
#------------------------------------------------------------------------------
# Uncomment to execute the Blender restart process
restart_blender()

# %%
#------------------------------------------------------------------------------
# Dataset Resizing and Scaling Function
#------------------------------------------------------------------------------
def resize_and_scale_dataset():
    """
    Resizes images and preserves corresponding labels for all dataset splits.
    
    This function:
    1. Resizes images from original size to new size
    2. Converts PNG images to JPG format
    3. Copies corresponding label files without modification
    4. Creates a configuration file for the scaled dataset
    """
    #--------------------------------------------------------------------------
    # Directory Structure Setup
    #--------------------------------------------------------------------------
    # Define directories structure for original and scaled datasets
    original_dirs = {
        split: [
            os.path.join(DEBRIS_DIR, split, "images"), 
            os.path.join(DEBRIS_DIR, split, "labels")
        ] for split in SPLITS
    }
    
    scaled_dirs = {
        split: [
            os.path.join(DEBRIS_SCALED_DIR, split, "images"), 
            os.path.join(DEBRIS_SCALED_DIR, split, "labels")
        ] for split in SPLITS
    }
    
    # Create output directories if they don't exist
    for dirs in scaled_dirs.values():
        os.makedirs(dirs[0], exist_ok=True)
        os.makedirs(dirs[1], exist_ok=True)
    
    #--------------------------------------------------------------------------
    # Image and Label Processing
    #--------------------------------------------------------------------------
    # Process each dataset split
    for split in SPLITS:
        image_orig_dir, label_orig_dir = original_dirs[split]
        image_scaled_dir, label_scaled_dir = scaled_dirs[split]
        
        # Process all PNG images in the original directory
        for img_path in glob.glob(f"{image_orig_dir}/*.png"):
            # Resize image and save as JPG
            img = cv2.imread(img_path)
            resized_img = cv2.resize(img, (NEW_SIZE, NEW_SIZE), interpolation=cv2.INTER_AREA)
            output_path = os.path.join(image_scaled_dir, os.path.basename(img_path).replace(".png", ".jpg"))
            cv2.imwrite(output_path, resized_img)
            print(f"Processed image: {output_path}")

            # Copy corresponding label file without modification
            # (No need to modify YOLO format labels as they use normalized coordinates)
            label_filename = os.path.basename(img_path).replace(".png", ".txt")
            label_path = os.path.join(label_orig_dir, label_filename)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(label_scaled_dir, label_filename))
                print(f"Copied label: {os.path.join(label_scaled_dir, label_filename)}")
    
    #--------------------------------------------------------------------------
    # Configuration File Creation
    #--------------------------------------------------------------------------
    # Create configuration file for the scaled dataset
    config_data = f"""path: {DEBRIS_SCALED_DIR}
train: train/images
val: val/images
test: test/images

nc: 4
names:
    0: Satellite
    1: Envisat
    2: Hubble
    3: Falcon 9 F&S
"""
    
    with open(CONFIG_PATH, "w") as f:
        f.write(config_data)
    
    print(f"Config file created at: {CONFIG_PATH}")

#------------------------------------------------------------------------------
# Function execution (uncomment to run)
#------------------------------------------------------------------------------
# Uncomment to execute the image scaling process
resize_and_scale_dataset()

# %%
#------------------------------------------------------------------------------
# Bounding Box Visualization Function
#------------------------------------------------------------------------------
def visualize_bounding_boxes():
    """
    Creates visualization of bounding box annotations by:
    1. Reading images and corresponding label files
    2. Drawing green bounding boxes based on the YOLO format annotations
    3. Adding red cross markers at the center of each bounding box
    4. Saving the annotated images to a separate directory
    
    This helps verify the correctness of annotations visually.
    """
    #--------------------------------------------------------------------------
    # Directory Setup
    #--------------------------------------------------------------------------
    # Create output directory structure for each split
    for split in SPLITS:
        out_split_dir = os.path.join(DEBRIS_ANNOTATED_DIR, split)
        os.makedirs(out_split_dir, exist_ok=True)
    
    #--------------------------------------------------------------------------
    # Image Processing Loop
    #--------------------------------------------------------------------------
    # Process each dataset split
    for split in SPLITS:
        images_dir = os.path.join(DEBRIS_DIR, split, "images")
        labels_dir = os.path.join(DEBRIS_DIR, split, "labels")
        out_split_dir = os.path.join(DEBRIS_ANNOTATED_DIR, split)
        
        # Process each image in the current split
        for filename in os.listdir(images_dir):
            # Check if the file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(images_dir, filename)
                
                # Get corresponding label filename
                base_name, _ = os.path.splitext(filename)
                label_file = os.path.join(labels_dir, base_name + ".txt")
                
                #--------------------------------------------------------------
                # Image Loading and Processing
                #--------------------------------------------------------------
                # Read the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error loading image: {image_path}")
                    continue

                height, width = image.shape[:2]
                
                #--------------------------------------------------------------
                # Label Processing and Visualization
                #--------------------------------------------------------------
                # Process the label file if it exists
                if os.path.exists(label_file):
                    with open(label_file, "r") as f:
                        for line in f:
                            # Parse YOLO format: class x_center y_center width height
                            parts = line.strip().split()
                            if len(parts) != 5:
                                print(f"Skipping invalid line in {label_file}: {line}")
                                continue
                            
                            # Convert normalized coordinates to pixel values
                            cls, x_center_norm, y_center_norm, w_norm, h_norm = parts
                            x_center = int(float(x_center_norm) * width)
                            y_center = int(float(y_center_norm) * height)
                            box_width = int(float(w_norm) * width)
                            box_height = int(float(h_norm) * height)
                            
                            # Calculate bounding box coordinates
                            x_min = int(x_center - box_width / 2)
                            y_min = int(y_center - box_height / 2)
                            x_max = int(x_center + box_width / 2)
                            y_max = int(y_center + box_height / 2)
                            
                            # Draw green bounding box
                            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 
                                        color=(0, 255, 0), thickness=1)
                            
                            # Draw red cross at the center
                            cross_size = 5
                            cv2.line(image, (x_center - cross_size, y_center), 
                                    (x_center + cross_size, y_center), 
                                    color=(0, 0, 255), thickness=1)
                            cv2.line(image, (x_center, y_center - cross_size), 
                                    (x_center, y_center + cross_size), 
                                    color=(0, 0, 255), thickness=1)
                else:
                    print(f"Label file not found for image: {filename}")
                
                #--------------------------------------------------------------
                # Save Annotated Image
                #--------------------------------------------------------------
                # Save the processed image with visualizations
                output_path = os.path.join(out_split_dir, filename)
                cv2.imwrite(output_path, image)
                print(f"Saved annotated image: {output_path}")

#------------------------------------------------------------------------------
# Function execution (uncomment to run)
#------------------------------------------------------------------------------
# Uncomment to execute the visualization process
visualize_bounding_boxes()

# %%
# ---------------------------------------------------------------------------
# Helper Functions to Read Images
# ---------------------------------------------------------------------------
def reading_image_albumentation(image_path):
    # Read image in BGR format using OpenCV
    read_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Convert BGR to RGB because albumentations expects an RGB image
    read_image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    return read_image

def reading_image_cv2(image_path):
    # Read image in BGR format using OpenCV (no conversion needed for cv2-based processing)
    read_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return read_image

# ---------------------------------------------------------------------------
# Augmentation Transformations
# ---------------------------------------------------------------------------
# Gaussian Noise Transform: adds Gaussian noise to an image.
Gaussian_Transform = A.GaussNoise(std_range=(0.08,0.08), mean_range=(0,0), p=1.0)
    # std_range, mean_range, and p are parameters/arguments of the GaussNoise function.
    # std_range: Specifies the variance range for the resulting Gaussian noise.
    # mean_range: Specifies the mean value for the noise.
    # p: Probability of applying the augmentation (1.0 means always applied).

# Random Occlusion Transform: randomly drops rectangular regions in the image.
Random_Occlusion_Transform = A.CoarseDropout(
    num_holes_range=(5,8),
    hole_height_range=(0.05,0.05),
    hole_width_range=(0.05,0.05),
    fill=(0,0,0),
    p=1.0
)
    # num_holes_range: Number of regions to drop out.
    # hole_height_range and hole_width_range: Size of holes, specified as a fraction of image dimensions.
    # fill: Color fill for the dropped regions (here, black).

# Motion Blur Transform: simulates motion blur effect.
Motion_Blur_Transform = A.MotionBlur(
    blur_limit=(3,3),
    angle_range=(0,360),
    direction_range=(-1.0,1.0),
    allow_shifted=True,
    p=1.0
)
    # blur_limit: Maximum kernel size for the blur effect.
    # angle_range: Possible angles (in degrees) for the motion blur.
    # direction_range: Direction control for how the blur extends.
    # allow_shifted: If True, allows random offset of the blur kernel.
    
# Grayscale Transform: converts an image to grayscale.
Grayscale_Transform = A.ToGray(p=1.0)

# Vignetting Transform: applies a vignette effect using Gaussian kernels.
def Vignetting_Transform(image, strength=0.5):
    rows, cols = image.shape[:2]
    # Create a Gaussian kernel for columns and rows
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols * strength)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows * strength)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()  # Normalize the mask to keep pixel values in range
    vignetting = (image * mask[..., np.newaxis]).astype(np.uint8)
    return vignetting

# Lens Distortion Transform: simulates lens distortion effects.
def Lens_Distortion_Transform(image, k1=-0.6, k2=0.2):
    h, w = image.shape[:2]
    # Define distortion coefficients and camera matrix
    dist_coeffs = np.array([k1, k2, 0, 0, 0])
    camera_matrix = np.array([[w, 0, w // 2], [0, h, h // 2], [0, 0, 1]], dtype="double")
    distorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    return distorted
    # k1 and k2 are the first two radial distortion coefficients.
    # Negative k1 typically creates a barrel distortion (bulging out), while positive k1 produces pincushion distortion (pinched edges).

# Poisson Noise Transform: simulates Poisson noise that is more visible in dark regions.
def Poisson_Noise_Transform(image, scale=0.1):
    image = image.astype(np.float32) / 255.0  # Normalize image to [0, 1] range.
    noisy = np.random.poisson(image * scale * 255.0) / (scale * 255.0)  # Apply Poisson noise.
    noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    return noisy.astype(np.uint8)  
    # Poisson noise effect depends on the image brightness, and lower scale values will result in stronger visible noise.

# Cosmic Ray Strikes Transform: simulates random bright spots due to cosmic ray strikes.
def Cosmic_Ray_Strikes_Transform(image, num_strikes=100):
    h, w, _ = image.shape  # Get image dimensions; ignore channel count (typically 3 for RGB).
    for _ in range(num_strikes):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        image = cv2.circle(image, (x, y), radius=np.random.randint(1, 2), 
                           color=(255, 255, 255), thickness=-1)
    return image
    # Each loop iteration selects a random coordinate and draws a small white circle to simulate a cosmic ray.

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
# Mapping augmentations to their application function and expected image type.
# For albumentations transforms, the type is "alb". For custom cv2 functions, the type is "cv2".
augmentations = {
    'gaussian': (Gaussian_Transform, 'alb'),
    'occlusion': (Random_Occlusion_Transform, 'alb'),
    'motion_blur': (Motion_Blur_Transform, 'alb'),
    'grayscale': (Grayscale_Transform, 'alb'),
    'vignetting': (Vignetting_Transform, 'cv2'),
    'lens_distortion': (Lens_Distortion_Transform, 'cv2'),
    'poisson': (Poisson_Noise_Transform, 'cv2'),
    'cosmic_rays': (Cosmic_Ray_Strikes_Transform, 'cv2')
}

def apply_augmentations():
    import time  # Import the time module to measure performance
    
    # Dictionary to store count for each augmentation
    augmentation_counts = {}
    
    # Dictionary to store total time for each augmentation
    augmentation_times = {}
    
    # Loop through each dataset split (e.g., train, val, test)
    # os.listdir(images_dir) returns the list of filenames in the specified directory.
    for split in SPLITS:
        images_dir = os.path.join(DEBRIS_SCALED_DIR, split, "images")
        labels_dir = os.path.join(DEBRIS_SCALED_DIR, split, "labels")
        
        # Process each image file in the directory
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_dir, filename)
                
                # Randomly select one augmentation from the dictionary
                aug_name, (aug_func, img_type) = random.choice(list(augmentations.items()))
                
                # Increment augmentation count
                augmentation_counts[aug_name] = augmentation_counts.get(aug_name, 0) + 1
                
                # Load the image based on the expected type of the augmentation
                if img_type == 'alb':
                    image = reading_image_albumentation(image_path)
                    
                    # Measure the time to apply the augmentation
                    start_time = time.time()
                    # Apply the albumentations augmentation, which returns a dict with key 'image'
                    augmented = aug_func(image=image)
                    augmented_image = augmented['image']
                    # Record the elapsed time
                    elapsed_time = time.time() - start_time
                    
                    # Convert the image back to BGR for saving with cv2
                    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                else:  # For cv2-based custom functions
                    image = reading_image_cv2(image_path)
                    
                    # Measure the time to apply the augmentation
                    start_time = time.time()
                    augmented_image = aug_func(image)
                    # Record the elapsed time
                    elapsed_time = time.time() - start_time
                
                # Add the elapsed time to the total time for this augmentation
                augmentation_times[aug_name] = augmentation_times.get(aug_name, 0) + elapsed_time
                
                # Create a new filename by appending the augmentation name before the file extension
                base, ext = os.path.splitext(filename)
                new_filename = f"{base}_{aug_name}{ext}"
                new_image_path = os.path.join(images_dir, new_filename)
                
                # Save the augmented image using cv2.imwrite
                cv2.imwrite(new_image_path, augmented_image)
                
                # Handle the associated label file, if it exists:
                # Copy the original label file to a new file with an updated name.
                original_label = os.path.join(labels_dir, base + ".txt")
                new_label = os.path.join(labels_dir, f"{base}_{aug_name}.txt")
                if os.path.exists(original_label):
                    shutil.copyfile(original_label, new_label)
                
                print(f"Augmented {filename} with {aug_name} and saved as {new_filename}")
    
    # After processing, print summary of augmentation counts and average times
    print("\nAugmentation Summary:")
    for aug, count in augmentation_counts.items():
        # Calculate the average time if the augmentation was applied at least once
        avg_time = augmentation_times.get(aug, 0) / count if count > 0 else 0
        print(f"  {aug}: {count} image(s) augmented, avg time: {avg_time:.6f} seconds per image")
    
    # Print overall average time across all augmentations
    total_images = sum(augmentation_counts.values())
    total_time = sum(augmentation_times.values())
    overall_avg_time = total_time / total_images if total_images > 0 else 0
    print(f"\nOverall average augmentation time: {overall_avg_time:.6f} seconds per image")

# ---------------------------------------------------------------------------
# Function Execution
# ---------------------------------------------------------------------------
# Uncomment the following line to execute the augmentation process.
apply_augmentations()

# %%
# ------------------------------------------------------------------------------
# Setup and Utility Functions for W&B, GPU, Training, and Logging
# ------------------------------------------------------------------------------

def setup_wandb() -> None:
    """
    Set up Weights & Biases integration with API key.
    
    - Configures the WANDB_API_KEY as an environment variable.
    - Prints project configuration details.
    """
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    print(f"W&B configured with project: {WANDB_PROJECT_NAME}")


def check_gpu_availability() -> None:
    """
    Check and print GPU details:
    
    - CUDA availability
    - Number of GPUs
    - GPU device name
    - Instructions for finding GPU index via nvidia-smi.
    """
    print("CUDA Available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("Number of GPUs detected:", torch.cuda.device_count())
        print("GPU Name Index 0:", torch.cuda.get_device_name(0))
        print("To find GPU Index, use nvidia-smi in command prompt")
    else:
        print("No GPU available, using CPU")
    
    print()  # Empty line for better readability


def train_yolov8_model(
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    device: int = 0
) -> Tuple[YOLO, str]:
    """
    Train a YOLOv8 model with W&B integration.
    
    Args:
        epochs: Number of training epochs.
        batch_size: Number of images per batch.
        image_size: Input image resolution.
        device: GPU device index (0 for first GPU).
    
    Returns:
        Tuple containing the trained model and the output directory path.
    """
    global run_name

    try:
        # Initialize model and display its info
        model = YOLO(MODEL_TYPE)
        model.info()
        print()
        
        # Create a unique run name with timestamp
        timestamp = datetime.now().strftime("%b-%d-%Y , %H-%M-%S")
        run_name = f"{RUN_NUMBER} {os.path.basename(MODEL_TYPE).split('.')[0]} ({timestamp})"
        
        # Initialize W&B run with the configured project and unique run name
        wandb.init(project=WANDB_PROJECT_NAME, name=run_name)
        
        print(f"Starting training with {epochs} epochs, batch size {batch_size}")
        model.train(
            data=CONFIG_PATH,
            epochs=epochs,
            device=device,
            imgsz=image_size,
            batch=batch_size,
            project=DEBRIS_SCALED_DIR,
            name=run_name,
            save_period=5  # Save checkpoint every 5 epochs
        )
        
        # Define run directory based on the unique run name
        run_dir = os.path.join(DEBRIS_SCALED_DIR, run_name)
        print(f"Training completed. Results saved to: {run_dir}")
        
        return model, run_dir
    except Exception as e:
        print(f"Error during training: {str(e)}")
        wandb.finish()
        raise


def extract_confusion_matrix(model: YOLO, run_dir: str) -> None:
    """
    Extract and save confusion matrix data from model validation.
    
    Args:
        model: Trained YOLO model instance.
        run_dir: Directory path for training outputs.
    
    Saves:
        - confusion_matrix_raw.csv: Raw confusion matrix data.
        - confusion_matrix_plot.png: Visualization of confusion matrix.
    """
    print("Running model validation to extract metrics...")
    
    validation_name = f"{os.path.basename(run_dir)} Validation"
    results = model.val(
        save_json=True,
        device=0,
        project=DEBRIS_SCALED_DIR,
        name=validation_name
    )
    
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        confusion_matrix = results.confusion_matrix.matrix
        
        # Convert confusion matrix to numpy array if on GPU (using .cpu())
        cm_data = confusion_matrix.cpu().numpy() if hasattr(confusion_matrix, 'cpu') else confusion_matrix
        
        cm_csv_path = os.path.join(run_dir, "confusion_matrix_raw.csv")
        pd.DataFrame(cm_data).to_csv(cm_csv_path)
        print(f"Raw confusion matrix data saved to: {cm_csv_path}")
        
        # Log raw confusion matrix data to W&B as a table
        wandb.log({"confusion_matrix_raw": wandb.Table(dataframe=pd.DataFrame(cm_data))})
        
        # Plot confusion matrix using matplotlib for visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_data, interpolation='nearest')
        plt.colorbar()
        plt.title('Confusion Matrix')
        
        cm_fig_path = os.path.join(run_dir, "confusion_matrix_plot.png")
        plt.savefig(cm_fig_path)
        plt.close()
        
        # Log the confusion matrix plot image to W&B
        wandb.log({"confusion_matrix_plot": wandb.Image(cm_fig_path)})
        print(f"Confusion matrix visualization saved to: {cm_fig_path}")
    else:
        print("No confusion matrix data available from validation.")


def log_existing_results_to_wandb(results_dir: str) -> None:
    """
    Upload existing training results to Weights & Biases.
    
    Args:
        results_dir: Path to directory containing:
            - results.csv: Training metrics.
            - weights/best.pt: Best model weights.
            - results.png: Training results plot.
    """
    print(f"Uploading existing results from: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return
    
    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"Results CSV not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Update W&B configuration with source file and the number of epochs (rows in CSV)
    wandb.config.update({
        "source_file": csv_path,
        "epochs": len(df)
    })
    
    # Log each epoch's metrics to W&B
    for index, row in df.iterrows():
        epoch = index + 1
        metrics = row.to_dict()
        wandb.log(metrics, step=epoch)
    
    # Log best model weights as a W&B artifact if available
    model_path = os.path.join(results_dir, "weights", "best.pt")
    if os.path.exists(model_path):
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        print("Model artifact logged.")
    
    # Log training results plot to W&B if exists
    results_plot_path = os.path.join(results_dir, "results.png")
    if os.path.exists(results_plot_path):
        wandb.log({"training_results": wandb.Image(results_plot_path)})
        print("Results plot logged.")

    print("Existing results successfully uploaded to W&B.")


def train_and_log_model(epochs=100, batch_size=16):
    """
    Complete pipeline for YOLOv8 model training with W&B integration.
    
    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
    """
    # Setup W&B and check for GPU availability
    setup_wandb()
    check_gpu_availability()
    
    # Train the YOLOv8 model and obtain the run directory
    model, run_dir = train_yolov8_model(epochs=epochs, batch_size=batch_size)
    # Extract and log the confusion matrix from model validation
    extract_confusion_matrix(model, run_dir)
    # Log existing results (CSV, plot, and best model artifact) to W&B
    log_existing_results_to_wandb(run_dir)
    
    # Finalize the W&B run
    wandb.finish()
    print("Training and logging completed successfully.")

# ------------------------------------------------------------------------------
# Function Execution
# ------------------------------------------------------------------------------
# Uncomment the following line to execute the training process.
train_and_log_model(epochs=100, batch_size=16)

# %%
# ------------------------------------------------------------------------------
# Inference Function for YOLOv8 Model
# ------------------------------------------------------------------------------
def run_inference(model_weight_path=None, confidence_threshold=0.80):
    """
    Perform inference using a trained YOLOv8 model on test images.
    
    Args:
        model_weight_path: Path to model weights. If None, uses a predefined path.
        confidence_threshold: Confidence threshold for detections (0-1).
    
    Creates timestamped results in the inference directory.
    """
    # --------------------------------------------------------------------------
    # Create a Timestamped Run Name
    # --------------------------------------------------------------------------
    # Generate a timestamp for the current inference run to keep outputs unique.
    datetime_today = datetime.now().strftime("%b-%d-%Y , %H-%M-%S")
    model_name = f"Inference {RUN_NUMBER} {MODEL_TYPE.replace(".pt","_best_weights")} ({datetime_today})"
    
    # --------------------------------------------------------------------------
    # Setup Directories
    # --------------------------------------------------------------------------
    # Define the directory containing test images and create an output directory for inference results.
    test_images_dir = os.path.join(DEBRIS_SCALED_DIR, "test", "images")
    # test_image_dir = r"C:\Users\Jai\Downloads\Individual Project Folder\yolov8-project\dataset6\Debris_Scaled\test\images\Hubble_Real_Life_2.jpg"

    inference_output_dir = os.path.join(DEBRIS_SCALED_DIR, "test", "images", "inference")
    os.makedirs(inference_output_dir, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # Determine the Model Weight Path
    # --------------------------------------------------------------------------
    # Use the provided model path or default to a specific run's best weights.
    if model_weight_path is None:
        model_weight_path = os.path.join(
            DEBRIS_SCALED_DIR, 
            "Run 9 yolov8n (Mar-25-2025 , 09-49-56)", 
            "weights", 
            "best.pt"
        )
    
    print(f"Loading model from: {model_weight_path}")
    model = YOLO(model_weight_path)
    
    # --------------------------------------------------------------------------
    # Run Inference
    # --------------------------------------------------------------------------
    # Execute the inference process on all images in the test directory.
    print(f"Running inference on images in: {test_images_dir}")
    results = model(
        source=test_images_dir, 
        show=False, 
        conf=confidence_threshold, 
        save=True, 
        project=inference_output_dir, 
        device=0, 
        save_conf=True,
        show_labels=True, 
        show_boxes=True, 
        show_conf=True,
        name=model_name
    )
    
    print(f"Inference completed. Results saved to: {os.path.join(inference_output_dir, model_name)}")
    return results

# ------------------------------------------------------------------------------
# Function Execution
# ------------------------------------------------------------------------------
# Uncomment the following line to execute the inference process
run_inference(confidence_threshold=0.60)

# %%
# ============================================================
# SET UP MATPLOTLIB CONFIGURATION & FONT
# ============================================================
# Specify full path to the Lato font file and add it
lato_regular_path = r"C:\Users\Jai\OneDrive - The University of Manchester\DESIGN\FONTS\Lato-Regular.ttf"
lato_bold_path = r"C:\Users\Jai\OneDrive - The University of Manchester\DESIGN\FONTS\Lato-Bold.ttf"
lato_italic_path = r"C:\Users\Jai\OneDrive - The University of Manchester\DESIGN\FONTS\Lato-Italic.ttf"

fm.fontManager.addfont(lato_regular_path)
fm.fontManager.addfont(lato_bold_path)
fm.fontManager.addfont(lato_italic_path)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Lato']

# ============================================================
# UTILITY FUNCTION
# ============================================================
def save_to_csv(folder_path, filename, data_dict):
    """
    Save data from a dictionary to a CSV file.
    
    Args:
        folder_path (str): Directory where the CSV will be saved.
        filename (str): Name of the CSV file.
        data_dict (dict): Dictionary with keys as column headers and values as data lists.
    """
    df = pd.DataFrame(data_dict)
    csv_path = os.path.join(folder_path, filename)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")

# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================
def load_coordinates(folder_path):
    """
    Load 3D coordinates from .txt files in the given folder.
    
    Each file should contain at least three numbers (X, Y, Z). If there are extra
    tokens, the last three are used as coordinates.
    
    Args:
        folder_path (str): Directory containing the .txt files.
    
    Returns:
        tuple: Lists of x, y, and z coordinate values.
    """
    x_vals, y_vals, z_vals = [], [], []
    file_count = 0
    parsed_count = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_count += 1
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r") as f:
                    content = f.read().strip()
                    tokens = content.split()
                    if len(tokens) >= 3:
                        # Use the last three tokens as coordinates
                        x = float(tokens[-3])
                        y = float(tokens[-2])
                        z = float(tokens[-1])
                        x_vals.append(x)
                        y_vals.append(y)
                        z_vals.append(z)
                        parsed_count += 1
                        # Print the first few parsed coordinates for debugging
                        if parsed_count <= 3:
                            print(f"Parsed {file_name}: {x}, {y}, {z}")
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
    
    print(f"Total coordinate files found: {file_count}")
    print(f"Successfully parsed coordinates: {parsed_count}")

    save_to_csv(folder_path, "coordinates.csv", {"x": x_vals, "y": y_vals, "z": z_vals})
    return x_vals, y_vals, z_vals

def load_debris_quaternion_coordinates(folder_path):
    """
    Load quaternions from .txt files in the given folder.
    
    Each file should contain at least four numbers (W, X, Y, Z). If there are extra
    tokens, the last four are used as coordinates.
    
    Args:
        folder_path (str): Directory containing the .txt files.
    
    Returns:
        tuple: Lists of w, x, y, and z Quaternion values.
    """
    w_vals, x_vals, y_vals, z_vals = [], [], [], []
    file_count = 0
    parsed_count = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_count += 1
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r") as f:
                    content = f.read().strip()
                    tokens = content.split()
                    if len(tokens) >= 4:
                        # Use the last four tokens as coordinates
                        w = float(tokens[-4])
                        x = float(tokens[-3])
                        y = float(tokens[-2])
                        z = float(tokens[-1])
                        
                        w_vals.append(w)
                        x_vals.append(x)
                        y_vals.append(y)
                        z_vals.append(z)

                        parsed_count += 1
                        # Print the first few parsed coordinates for debugging
                        if parsed_count <= 3:
                            print(f"Parsed {file_name}: {w}, {x}, {y}, {z}")
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
    
    print(f"Total quaternion files found: {file_count}")
    print(f"Successfully parsed quaternions: {parsed_count}")

    save_to_csv(folder_path, "debris_quaternion.csv", {"w": w_vals, "x": x_vals, "y": y_vals, "z": z_vals})
    return w_vals, x_vals, y_vals, z_vals

def load_offset_values(folder_path):
    """
    Load offset values from .txt files in the provided folder.
    
    Frame numbers are extracted from the filenames (first number) and the file content
    is expected to contain a single offset value.
    
    Args:
        folder_path (str): Directory with offset .txt files.
    
    Returns:
        tuple: Sorted lists of frames and their corresponding offset values.
    """
    offsets = {}
    file_count = 0
    parsed_count = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_count += 1
            file_path = os.path.join(folder_path, file_name)
            try:
                # Extract the first number from the filename as the frame number
                match = re.search(r'(\d+)', file_name)
                if match:
                    frame_num = int(match.group(0))
                    with open(file_path, "r") as f:
                        content = f.read().strip()
                        offset = float(content)
                        offsets[frame_num] = offset
                        parsed_count += 1
                        if parsed_count <= 3:
                            print(f"Parsed {file_name}: Frame {frame_num} offset {offset}")
            except Exception as e:
                print(f"Error parsing offset file {file_name}: {e}")
    
    print(f"Total offset files found: {file_count}")
    print(f"Successfully parsed offsets: {parsed_count}")

    # Sort results by frame number
    sorted_frames = sorted(offsets.keys())
    sorted_offsets = [offsets[frame] for frame in sorted_frames]

    save_to_csv(folder_path, "offsets.csv", {"frame": sorted_frames, "offset": sorted_offsets})
    return sorted_frames, sorted_offsets

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def plot_3d_scatter(x_vals, y_vals, z_vals, title='Debris Coordinates', marker_size=20, alpha=0.7):
    """
    Create a 3D scatter plot of the provided coordinates.
    
    Args:
        x_vals (list): List of x coordinate values.
        y_vals (list): List of y coordinate values.
        z_vals (list): List of z coordinate values.
        title (str): Title of the plot.
        marker_size (int): Size of the markers.
        alpha (float): Transparency level of the markers.
    
    Returns:
        matplotlib.figure.Figure: Figure containing the 3D scatter plot.
    """
    BG_IMAGE = plt.imread(r"C:\Users\Jai\Downloads\Individual Project Folder\yolov8-project\dataset6\BG_IMAGE2.png")

    # Create figure
    fig = plt.figure(figsize=(10, 10))

    # Create background axes that span the entire figure and display the image
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.imshow(BG_IMAGE, aspect='auto')
    ax_bg.axis('off')

    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection='3d')
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # Plot scatter points with black edges for better visibility
    ax.scatter(x_vals, y_vals, z_vals, c='gray', s=marker_size, alpha=alpha, edgecolors='k')

    # Set axis labels and plot title
    ax.set_xlabel('X', labelpad=10, fontsize=20, fontweight='bold', color='black')
    ax.set_ylabel('Y', labelpad=10, fontsize=20, fontweight='bold', color='black')
    ax.set_zlabel('Z', labelpad=10, fontsize=20, fontweight='bold', color='black')
    ax.set_title(title, pad=0, fontsize=34, color='black', fontstyle='italic')
    ax.grid(True)

    # Set equal aspect ratio for all axes
    max_range = np.array([max(x_vals)-min(x_vals), max(y_vals)-min(y_vals), max(z_vals)-min(z_vals)]).max() / 2.0
    mid_x = (max(x_vals) + min(x_vals)) * 0.5
    mid_y = (max(y_vals) + min(y_vals)) * 0.5
    mid_z = (max(z_vals) + min(z_vals)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Make tick labels bold 
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_zticklabels():
        label.set_fontweight('bold')

    return fig

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def plot_3d_quaternion_scatter(w_vals, x_vals, y_vals, z_vals, title='3D Quaternion Scatter', marker_size=20, alpha=0.7):
    """
    Create a 3D scatter plot of quaternion orientations.
    
    Args:
        w_vals (list): List of w quaternion values.
        x_vals (list): List of x quaternion values.
        y_vals (list): List of y quaternion values.
        z_vals (list): List of z quaternion values.
        title (str): Title of the plot.
        marker_size (int): Size of the markers.
        alpha (float): Transparency level of the markers.
    
    Returns:
        matplotlib.figure.Figure: Figure containing the 3D scatter plot.
    """
    BG_IMAGE = plt.imread(r"C:\Users\Jai\Downloads\Individual Project Folder\yolov8-project\dataset6\BG_IMAGE2.png")

    # Create figure
    fig = plt.figure(figsize=(10, 10))

    # Create background axes that span the entire figure and display the image
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.imshow(BG_IMAGE, aspect='auto')
    ax_bg.axis('off')

    # Create 3D plot
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection='3d')
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Convert quaternions to rotation vectors (for visualization)
    quaternions = np.column_stack((w_vals, x_vals, y_vals, z_vals))
    rotation_vectors = R.from_quat(quaternions).as_rotvec()
    
    # Extract X, Y, Z components
    x, y, z = rotation_vectors[:, 0], rotation_vectors[:, 1], rotation_vectors[:, 2]

    # Plot scatter points with black edges for better visibility
    ax.scatter(x, y, z, c='gray', s=marker_size, alpha=alpha, edgecolors='k')

    # Set axis labels and plot title
    ax.set_xlabel('X', labelpad=10, fontsize=20, fontweight='bold', color='black')
    ax.set_ylabel('Y', labelpad=10, fontsize=20, fontweight='bold', color='black')
    ax.set_zlabel('Z', labelpad=10, fontsize=20, fontweight='bold', color='black')
    ax.set_title(title, pad=1, fontsize=34, color='black', fontstyle='italic')
    ax.grid(True)

    # Set equal aspect ratio for all axes
    max_range = np.array([max(x)-min(x), max(y)-min(y), max(z)-min(z)]).max() / 2.0
    mid_x = (max(x) + min(x)) * 0.5
    mid_y = (max(y) + min(y)) * 0.5
    mid_z = (max(z) + min(z)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Make tick labels bold 
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_zticklabels():
        label.set_fontweight('bold')

    return fig

def plot_euler_angle_scatter(roll_vals, pitch_vals, yaw_vals, title='Scatter Plot of Euler Angles', marker_size=20, alpha=0.7):
    """
    Create a 2D scatter plot for Roll vs Pitch, Roll vs Yaw, and Pitch vs Yaw,
    with a background image behind the main plot.

    Args:
        roll_vals (list): List of Roll values.
        pitch_vals (list): List of Pitch values.
        yaw_vals (list): List of Yaw values.
        title (str): Title of the plot.
        marker_size (int): Size of the markers.
        alpha (float): Transparency level of the markers.

    Returns:
        matplotlib.figure.Figure: Figure containing the 2D scatter plot.
    """
    BG_IMAGE = plt.imread(r"C:\Users\Jai\Downloads\Individual Project Folder\yolov8-project\dataset6\BG_IMAGE2.png")

    # Create the figure (no default axes yet)
    fig = plt.figure(figsize=(10, 8))

    # --- Background Axes ---
    # Spans the entire figure; placed behind the main axis (zorder=0).
    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax_bg.imshow(BG_IMAGE, aspect='auto')
    ax_bg.axis('off')  # Hide ticks on the background axis

    # --- Main Plot Axes ---
    # Add the main axis on top (zorder=1)
    ax = fig.add_subplot(111, zorder=1)
    ax.set_facecolor('white')  # Transparent so background is visible
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

    # Scatter plots with different colors and markers
    ax.scatter(roll_vals, pitch_vals, c='red', alpha=alpha, marker='x', 
               s=marker_size, label="Roll vs Pitch")
    ax.scatter(roll_vals, yaw_vals, c='green', alpha=alpha, marker='x', 
               s=marker_size, label="Roll vs Yaw")
    ax.scatter(pitch_vals, yaw_vals, c='blue', alpha=alpha, marker='^', 
               s=marker_size, label="Pitch vs Yaw")

    # Set axis labels and plot title
    ax.set_xlabel("Angle (radians)", fontsize=20, fontweight="bold", color="black")
    ax.set_ylabel("Angle (radians)", fontsize=20, fontweight="bold", color="black")
    ax.set_title(title, va='bottom', fontsize=34, color="black", fontweight="bold", pad=15, fontstyle='italic')

    # Legend
    ax.legend(fontsize=12, loc="upper right")

    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    return fig

def plot_offset_values(frames, offsets, title='Offset Values', color='gray',xlabel="Enter Axis Title",ylabel="Enter Axis Title"):
    """
    Create a polar plot of offset values against frame numbers.
    
    Frame numbers are normalized to angles over a full circle (0 to 2π).
    The radial axis shows the offset values, and a box is drawn around the plot area.
    
    Args:
        frames (list): List of frame numbers.
        offsets (list): List of corresponding offset values.
        title (str): Plot title.
        color (str): Color for the markers.
    
    Returns:
        matplotlib.figure.Figure: Figure containing the polar plot.
    """
    COLOR_BG = 'white'
    COLOR_ANGULAR_AXIS_LABELS = 'black'
    COLOR_RADIAL_AXIS_LABELS = 'black'
    COLOR_AXIS_TITLE = 'black'
    COLOR_TITLE = 'black'
    BG_IMAGE = plt.imread(r"C:\Users\Jai\Downloads\Individual Project Folder\yolov8-project\dataset6\BG_IMAGE2.png")

    # Create figure
    fig = plt.figure(figsize=(8, 8))

    # Create background axes that span the entire figure and display the image
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.imshow(BG_IMAGE, aspect='auto')
    ax_bg.axis('off')

    # Create polar axes on top with transparent background
    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor('white')

    # Normalize frame numbers to angles in radians
    min_frame, max_frame = min(frames), max(frames)
    theta = [2 * np.pi * ((frame - min_frame) / (max_frame - min_frame)) for frame in frames]

    # Plot only markers (without connecting line)
    ax.plot(theta, offsets, marker='o', linestyle='', color=color, markeredgecolor='black', markeredgewidth = 0.5, zorder=1)

    # Set radial limits and tick marks (offset values from 0 to 100)
    ax.set_rlim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_rlabel_position(90)
    ax.tick_params(axis='y', labelsize=14)

    # Set angular tick marks based on normalized frame numbers
    num_ticks = 7
    frame_ticks = np.linspace(min_frame, max_frame, num=num_ticks)
    theta_ticks = [2 * np.pi * ((frame - min_frame) / (max_frame - min_frame)) for frame in frame_ticks]
    labels = [str(int(frame + 1)) for frame in frame_ticks]
    labels[0] = ''
    labels[-1] = ''
    ax.set_thetagrids(np.degrees(theta_ticks), labels=labels, color=COLOR_ANGULAR_AXIS_LABELS, fontsize=14, fontweight = 'bold')

    r_max = ax.get_rmax() 
    for i in range(num_ticks - 1):
        if i % 2 == 1:  
            theta_start = theta_ticks[i]               
            width = theta_ticks[i+1] - theta_ticks[i]     # Angular width of the sector
            # Draw a bar (sector) from radius 0 to r_max with the given width.
            ax.bar(theta_start, r_max, width=width, bottom=0,
                color='#c4c4c4', alpha=0.3, edgecolor='none', zorder=0, align='edge')

    # Style the radial (offset) tick labels
    for label in ax.get_yticklabels():
        label.set_zorder(10)
        label.set_fontweight('bold')
        label.set_color(COLOR_RADIAL_AXIS_LABELS)
        label.set_horizontalalignment('center')

    # Add text annotation showing the frame range along the outer edge
    angle = 0  # Corresponds to 0 degrees
    r_out = ax.get_rmax() * 1.115
    ax.text(angle, r_out, f"{int(frame_ticks[0])}, {int(frame_ticks[-1] + 1)}",
            ha='center', va='center', color=COLOR_ANGULAR_AXIS_LABELS, fontsize=14, fontweight='bold')

    # Place the plot title above the plot area
    ax.set_title(title, va='bottom', pad=20, fontsize=34, color=COLOR_TITLE, fontstyle='italic')

    # Add custom axis titles as annotations (they sit outside the main plot box)
    ax.text(np.radians(180), ax.get_rmax() * 1.2, ylabel,
            ha='center', va='center', fontsize=20, color=COLOR_AXIS_TITLE, fontweight='bold', rotation=90)
    ax.text(np.radians(270), ax.get_rmax() * 1.1, xlabel,
            ha='center', va='center', fontsize=20, color=COLOR_AXIS_TITLE,fontweight='bold')

    ax.grid(True)

    return fig

def plot_augmentation_summary(summary_file_path, title='Augmentation Summary', bar_color='gray'):
    """
    Read an augmentations summary text file and create a bar chart.
    
    The text file is expected to have a header line followed by lines in the format:
        augmentation_name: count image(s) augmented
    This function will ignore the header line and reformat augmentation names
    (e.g., "motion_blur" becomes "Motion Blur") for the x-axis labels.
    
    The bar chart uses the same background image and formatting as the other plots.
    
    Args:
        summary_file_path (str): Full path to the Augmentations Summary text file.
        title (str): Title of the bar chart.
        bar_color (str): Color for the bars.
    
    Returns:
        matplotlib.figure.Figure: Figure containing the bar chart.
    """
    import re
    # Load the common background image
    BG_IMAGE = plt.imread(r"C:\Users\Jai\Downloads\Individual Project Folder\yolov8-project\dataset6\BG_IMAGE2.png")
    
    # Create figure and background axes
    fig = plt.figure(figsize=(10, 6))
    ax_bg = fig.add_axes([0, 0, 1, 1])
    #ax_bg.set_facecolor('#e1f2f5')
    ax_bg.imshow(BG_IMAGE, aspect='auto')

    ax_bg.axis('off')
    
    # Create main axes with white background and grid styling
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    
    augmentation_names = []
    counts = []
    
    with open(summary_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines if present
            if not line or line.lower().startsWith("augmentation summary"):
                continue
            # Expecting a colon to separate the augmentation name and the count info
            if ':' in line:
                parts = line.split(':', 1)
                raw_name = parts[0].strip()
                # Reformat name: replace underscores with spaces and title-case it
                name = raw_name.replace('_', ' ').title()
                # Extract the first number from the remainder of the line
                match = re.search(r'\d+', parts[1])
                count = int(match.group()) if match else 0
                augmentation_names.append(name)
                counts.append(count)
    
    # List of pastel colours
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(augmentation_names)))

    # Create bar chart
    ax.bar(augmentation_names, counts, color=colors, edgecolor='black', linewidth=0.5)

    # Replace space in augmentation names with a new-line character
    modified_labels = []
    for label in augmentation_names:
        label = label.replace(' ','\n')
        modified_labels.append(label)
    
    # Set title and axis labels with consistent formatting
    ax.set_xlabel("Augmentation Type", fontsize=20, fontweight="bold", color="black", labelpad=5)
    ax.set_ylabel("Augmentation Count", fontsize=20, fontweight="bold", color="black", labelpad=10)
    ax.set_title(title, fontsize=34, fontweight="bold", fontstyle="italic", color="black", pad=20)
    ax.grid(False)
    
    # Enhance tick label appearance
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_fontweight("bold")
    ax.set_xticks(range(len(modified_labels)))
    ax.set_xticklabels(modified_labels, rotation=45, ha='right')
    
    fig.subplots_adjust(bottom=0.4)

    return fig

def confusion_matrices():
    # Define the list of confusion matrix file paths and corresponding model names
    confusion_matrices = [CONFUSION_MATRIX_8, CONFUSION_MATRIX_11, CONFUSION_MATRIX_12]
    model_names = ["YOLOv8n", "YOLOv11n", "YOLOv12n"]

    # Loop through each confusion matrix
    for i, confusion_matrix in enumerate(confusion_matrices):
        # Read the confusion matrix data from the CSV file
        # The first column is used as the row index
        confusion_df = pd.read_csv(confusion_matrix, index_col=0)

        # Convert the DataFrame to a NumPy array and ensure values are integers
        conf_matrix = confusion_df.values.astype(int)

        # Define class labels in the order they appear in the matrix
        class_labels = ["Satellite", "Envisat", "Hubble", "Falcon 9 F&S"]

        # Create a new figure with specific dimensions (width, height in inches)
        plt.figure(figsize=(8, 6))

        # Select a specific color for each model for visual distinction
        if i == 0:
            hex_color = "#13E7C4"  # Teal color for YOLOv8n
        elif i == 1:
            hex_color = "#E97132"  # Orange color for YOLOv11n
        else:
            hex_color = "#196B24"  # Green color for YOLOv12n
        
        # Create a custom color gradient from white to the model's color
        cmap = LinearSegmentedColormap.from_list('custom_color', ['white', hex_color])

        # Generate the heatmap visualization
        ax = sns.heatmap(
            conf_matrix,          # The confusion matrix data
            annot=True,           # Show the values in each cell
            fmt='d',              # Format values as integers
            cmap=cmap,            # Use our custom color gradient
            xticklabels=class_labels,  # Labels for the x-axis (predicted classes)
            yticklabels=class_labels   # Labels for the y-axis (actual classes)
        )

        # Add title and axis labels with custom formatting
        plt.title(f"Confusion Matrix - {model_names[i]}", fontsize=24, fontweight="bold", pad=20)
        plt.xlabel("Predicted Label", fontsize=16, fontweight="bold", labelpad=10)
        plt.ylabel("True Label", fontsize=16, fontweight="bold", labelpad=10)

        # Format the tick labels to be bold and horizontal
        plt.xticks(fontweight="bold", rotation=0)
        plt.yticks(fontweight="bold", rotation=0)

        # Adjust the layout to ensure all elements fit within the figure
        plt.tight_layout()

        # Save the figure to disk with a model-specific filename
        save_path = os.path.join(DATASET_BASE,rf"Results\confusion_matrix_{model_names[i]}.png")
        plt.savefig(save_path)

# ============================================================
# MAIN EXECUTION BLOCK
# ============================================================
if __name__ == "__main__":
    # Define folders for data sources
    debris_folder = r"C:\Users\Jai\Downloads\Individual Project Folder\yolov8-project\dataset5\orbit_container_coordinates"
    container_folder = r"C:\Users\Jai\Downloads\Individual Project Folder\yolov8-project\dataset5\debris_tracking_container_coordinates"
    camera_folder = r"C:\Users\Jai\Downloads\Individual Project Folder\yolov8-project\dataset5\camera_container_coordinates"

    # ------------------------------------------------------------
    # Process and plot 3D coordinates for debris
    # ------------------------------------------------------------
    print("\n===== PROCESSING DEBRIS COORDINATES =====")
    x_vals, y_vals, z_vals = load_coordinates(ORBIT_CONTAINER_COORDINATES)
    debris_fig = plot_3d_scatter(x_vals, y_vals, z_vals,
                                 title='Debris position relative to Earth',
                                 marker_size=15, alpha=0.8)

    # ------------------------------------------------------------
    # Process and plot quaternions for debris orientation
    # ------------------------------------------------------------
    print("\n===== PROCESSING DEBRIS ORIENTATIONS QUATERNIONS =====")
    w_vals, x_vals, y_vals, z_vals = load_debris_quaternion_coordinates(QUATERNION_COORDINATES)
    debris_quaternion_fig = plot_3d_quaternion_scatter(w_vals, x_vals, y_vals, z_vals,
                                                       title='Debris Orientation (Quaternion)',
                                                       marker_size=15, alpha=0.8)
    
    # ------------------------------------------------------------
    # Process and plot Euler for debris orientation
    # ------------------------------------------------------------
    print("\n===== PROCESSING DEBRIS ORIENTATIONS EULER =====")
    roll_vals, pitch_vals, yaw_vals = load_coordinates(EULER_COORDINATES)
    debris_euler_fig = plot_euler_angle_scatter(roll_vals, pitch_vals, yaw_vals,
                                                title='Debris Orientation (Euler)',
                                                marker_size=15, alpha=0.8)

    # ------------------------------------------------------------
    # Process and plot offset values for container
    # ------------------------------------------------------------
    print("\n===== PROCESSING CONTAINER OFFSETS =====")
    container_frames, container_offsets = load_offset_values(DEBRIS_TRACKING_CONTAINER_COORDINATES)
    container_fig = plot_offset_values(container_frames, container_offsets,
                                       title='Camera position relative to Debris',
                                       color='gray',
                                       ylabel="Position on 'Debris Tracking Geometry'",
                                       xlabel="Image Number")

    # ------------------------------------------------------------
    # Process and plot offset values for camera
    # ------------------------------------------------------------
    print("\n===== PROCESSING CAMERA OFFSETS =====")
    camera_frames, camera_offsets = load_offset_values(CAMERA_CONTAINER_COORDINATES)
    camera_fig = plot_offset_values(camera_frames, camera_offsets,
                                    title='Camera Offset Values',
                                    color='gray',
                                    ylabel="Position on 'Camera Geometry'",
                                    xlabel="Image Number")

    # ------------------------------------------------------------
    # Process and plot augmentation summary bar chart
    # ------------------------------------------------------------
    print("\n===== PROCESSING AUGMENTATION SUMMARY =====")
    augmentation_summary_file = AUGMENTATION_SUMMARY
    augmentation_fig = plot_augmentation_summary(augmentation_summary_file,
                                                 title="Augmentation Summary",
                                                 bar_color='gray')
    
    # ------------------------------------------------------------
    # Plot and save confusion matrix
    # ------------------------------------------------------------
    confusion_matrices()

    # ------------------------------------------------------------
    # Save plots as image files (with 300 dpi resolution)
    # ------------------------------------------------------------
    debris_fig.savefig(os.path.join(ORBIT_CONTAINER_COORDINATES, "Orbit Container Plot.png"), dpi=300, bbox_inches='tight')
    container_fig.savefig(os.path.join(DEBRIS_TRACKING_CONTAINER_COORDINATES, "Debris Tracking Container Plot.png"), dpi=300)
    camera_fig.savefig(os.path.join(CAMERA_CONTAINER_COORDINATES, "camera_container_plot.png"), dpi=300)
    debris_quaternion_fig.savefig(os.path.join(QUATERNION_COORDINATES, "Debris Orientation Quaternion.png"), dpi=300)
    debris_euler_fig.savefig(os.path.join(EULER_COORDINATES, "Debris Orientation Euler.png"), dpi=300)
    augmentation_fig.savefig(os.path.join(os.path.dirname(augmentation_summary_file), "Augmentation Summary Plot.png"), dpi=300)

    # Adjust layout and display all figures
    plt.tight_layout()
    plt.close()
