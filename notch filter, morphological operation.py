import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def fft_notch_filter(image, cutoff=30):
    """Apply a lightweight FFT notch filter to remove periodic noise."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Create a notch reject filter mask
    mask = np.ones((rows, cols), np.uint8)

    # Suppress low frequencies around the center (notch)
    r = cutoff
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    # Apply mask
    fshift_filtered = fshift * mask

    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize to 8-bit
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_back

def morphological_opening(image, kernel_size=1):
    """Apply morphological opening (erosion followed by dilation)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened

def process_dataset(input_main_folder, output_main_folder):
    for root, _, files in os.walk(input_main_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                # Step 1: FFT Notch filter
                fft_filtered = fft_notch_filter(img, cutoff=30)

                # Step 2: Morphological opening
                morph_opened = morphological_opening(fft_filtered, kernel_size=5)

                # ----------------------------
                # Save Morphological output
                # ----------------------------
                rel_path = os.path.relpath(root, input_main_folder)  # keep subfolder structure
                save_dir = os.path.join(output_main_folder, rel_path)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"fft_morph_{filename}")
                cv2.imwrite(save_path, morph_opened)

                print(f"✅ Saved: {save_path}")

                # ----------------------------
                # Display Input vs Output
                # ----------------------------
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title("Original")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(fft_filtered, cmap="gray")
                plt.title("FFT Notch Filtered")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(morph_opened, cmap="gray")
                plt.title("Morphological Opening")
                plt.axis("off")

                plt.show()

# ----------------------------
# Example Usage
# ----------------------------
input_main_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/guided_output"   # Input dataset
output_main_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/fft_morph_output"   # Output dataset
process_dataset(input_main_folder, output_main_folder)