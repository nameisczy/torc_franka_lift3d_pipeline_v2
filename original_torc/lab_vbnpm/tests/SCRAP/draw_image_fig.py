#!/usr/bin/env python3
"""
Script to read and display color, depth, and segmentation images in a compact matplotlib figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
from skimage.exposure import rescale_intensity
import colorcet as cc
from matplotlib.colors import ListedColormap

def load_color_image(image_path):
    """Load a color image from file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Color image not found: {image_path}")
    
    # Load image using OpenCV (BGR format) and convert to RGB
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load color image: {image_path}")
    
    # Convert BGR to RGB for proper matplotlib display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_depth_image(image_path):
    """Load a depth image from file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Depth image not found: {image_path}")
    
    # Load depth image - typically 16-bit single channel
    depth_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    if depth_image is None:
        raise ValueError(f"Failed to load depth image: {image_path}")

    # depth_image = rescale_intensity(
    #     depth_image, in_range=(450,1000), out_range=(0, 255)
    # ).astype(np.uint8)
    # depth_image2 = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    # cv2.imshow("Depth Image", depth_image2)

    return depth_image


def load_segmentation_image(npy_path):
    """Load a segmentation array from numpy file."""
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Segmentation file not found: {npy_path}")
    
    if not npy_path.endswith('.npy'):
        raise ValueError(f"Segmentation file must be a .npy file: {npy_path}")
    
    # Load segmentation array from numpy file
    seg_array = np.load(npy_path)
    
    # Ensure it's a 2D array
    if seg_array.ndim != 2:
        raise ValueError(f"Segmentation array must be 2D, got shape: {seg_array.shape}")
    
    return seg_array


def crop_bottom_half(image):
    """Crop the bottom half of an image."""
    height = image.shape[0]
    start_row = height // 2
    return image[start_row:, :]

def visualize_images(color_img, depth_img, seg_img, save_path=None):
    """
    Display color, depth, and segmentation images in a compact matplotlib figure.
    
    Args:
        color_img (np.ndarray): RGB color image
        depth_img (np.ndarray): Depth image
        seg_img (np.ndarray): Segmentation image
        save_path (str, optional): Path to save the figure
    """
    # Crop all images to bottom half
    color_img_cropped = crop_bottom_half(color_img)
    depth_img_cropped = crop_bottom_half(depth_img)
    seg_img_cropped = crop_bottom_half(seg_img)
    
    # Create figure with subplots arranged vertically
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    
    # Display color image - center aligned
    axes[0].imshow(color_img_cropped)
    axes[0].axis('off')
    
    # Display depth image with colormap - center aligned
    axes[1].imshow(depth_img_cropped, cmap='CMRmap', vmin=400, vmax=1000)
    axes[1].axis('off')
    
    # Display segmentation image with discrete colormap - center aligned
    cmap = cc.glasbey_dark
    cmap[0]= '#ffffff'
    axes[2].imshow(seg_img_cropped, cmap=ListedColormap(cmap))
    axes[2].axis('off')
    
    # Adjust layout to make images compact and centered
    plt.tight_layout(pad=0.1)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Display the figure
    plt.show()


def main():
    """Main function to handle command line arguments and execute visualization."""
    parser = argparse.ArgumentParser(description='Visualize color, depth, and segmentation images')
    parser.add_argument('--color', required=True, help='Path to color image file')
    parser.add_argument('--depth', required=True, help='Path to depth image file') 
    parser.add_argument('--segmentation', required=True, help='Path to segmentation image file')
    parser.add_argument('--save', help='Path to save the output figure (optional)')
    
    args = parser.parse_args()
    
    try:
        # Load images
        print("Loading images...")
        color_img = load_color_image(args.color)
        depth_img = load_depth_image(args.depth)
        seg_img = load_segmentation_image(args.segmentation)
        
        print(f"Color image shape: {color_img.shape}")
        print(f"Depth image shape: {depth_img.shape}, range: [{np.min(depth_img)}, {np.max(depth_img)}]")
        print(f"Segmentation image shape: {seg_img.shape}, unique labels: {len(np.unique(seg_img))}")
        
        # Visualize images
        visualize_images(color_img, depth_img, seg_img, args.save)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


def demo_with_synthetic_data():
    """
    Demo function that creates synthetic data for testing when no real images are available.
    """
    print("Running demo with synthetic data...")
    
    # Create synthetic color image (RGB)
    height, width = 480, 640
    color_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some structure to make it more realistic
    color_img[100:200, 200:400] = [255, 0, 0]  # Red rectangle
    color_img[250:350, 300:500] = [0, 255, 0]  # Green rectangle
    
    # Create synthetic depth image
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    depth_img = (1000 + 500 * np.sin(x/50) * np.cos(y/50)).astype(np.uint16)
    
    # Create synthetic segmentation image
    seg_img = np.zeros((height, width), dtype=np.uint8)
    seg_img[100:200, 200:400] = 1  # Object 1
    seg_img[250:350, 300:500] = 2  # Object 2
    seg_img[50:150, 50:150] = 3    # Object 3
    
    # Visualize the synthetic images
    visualize_images(color_img, depth_img, seg_img)


if __name__ == "__main__":
    import sys
    
    # If no arguments provided, run demo
    if len(sys.argv) == 1:
        demo_with_synthetic_data()
    else:
        exit_code = main()
        sys.exit(exit_code)
