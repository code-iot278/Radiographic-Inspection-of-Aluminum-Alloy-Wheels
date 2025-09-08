import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def gamma_normalization(image, gamma=1.5):
    """Apply gamma correction (normalization)."""
    img_norm = image / 255.0
    img_gamma = np.power(img_norm, gamma)
    return np.uint8(img_gamma * 255)

def process_dataset(input_main_folder, output_main_folder):
    for root, _, files in os.walk(input_main_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                # Step 1: Gamma normalization
                gamma_corrected = gamma_normalization(img, gamma=1.5)

                # Step 2: Standardize to 8-bit depth
                if gamma_corrected.dtype != np.uint8:
                    gamma_corrected = cv2.normalize(
                        gamma_corrected, None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)

                # Step 3: Guided filter
                guided = cv2.ximgproc.guidedFilter(
                    guide=gamma_corrected, src=gamma_corrected,
                    radius=8, eps=0.04
                )

                # ----------------------------
                # Save Guided Filtered output
                # ----------------------------
                rel_path = os.path.relpath(root, input_main_folder)  # keep subfolder structure
                save_dir = os.path.join(output_main_folder, rel_path)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"guided_{filename}")
                cv2.imwrite(save_path, guided)

                print(f"✅ Saved: {save_path}")

                # ----------------------------
                # Display Input vs Output
                # ----------------------------
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title("Original")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB))
                plt.title("Gamma Normalized")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(cv2.cvtColor(guided, cv2.COLOR_BGR2RGB))
                plt.title("Guided Filtered")
                plt.axis("off")

                plt.show()

# ----------------------------
# Example Usage
# ----------------------------
input_main_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/merged_dataset"   # Input dataset
output_main_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/guided_output"   # Output dataset
process_dataset(input_main_folder, output_main_folder)
