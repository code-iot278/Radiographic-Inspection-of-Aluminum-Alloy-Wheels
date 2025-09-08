import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# 1. Main input folder path
# ----------------------------
main_input_folder = "/content/ChatGPT Image Sep 8, 2025, 10_36_31 AM.png"  # main folder with subfolders
output_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/cropped_wheels"
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# 2. Walk through all subfolders
# ----------------------------
for root, dirs, files in os.walk(main_input_folder):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, filename)

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Preprocess
            gray_blur = cv2.medianBlur(gray, 5)

            # Hough Circle detection
            circles = cv2.HoughCircles(
                gray_blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=100,
                param1=100,
                param2=40,
                minRadius=100,
                maxRadius=250
            )

            # ----------------------------
            # 3. Display input and output
            # ----------------------------
            plt.figure(figsize=(12, 6))

            # Input image
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Input Image")
            plt.axis("off")

            # Output wheel crop
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Ensure cropping stays inside image boundaries
                    y1, y2 = max(0, y-r), min(img.shape[0], y+r)
                    x1, x2 = max(0, x-r), min(img.shape[1], x+r)

                    wheel_crop = img[y1:y2, x1:x2]

                    if wheel_crop.size == 0:  # skip invalid crops
                        continue

                    plt.subplot(1, 2, 2)
                    plt.imshow(cv2.cvtColor(wheel_crop, cv2.COLOR_BGR2RGB))
                    plt.title("Separated Wheel")
                    plt.axis("off")

                    # Save cropped wheel in same subfolder structure
                    rel_path = os.path.relpath(root, main_input_folder)
                    save_dir = os.path.join(output_folder, rel_path)
                    os.makedirs(save_dir, exist_ok=True)

                    save_path = os.path.join(save_dir, f"wheel_{filename}")
                    cv2.imwrite(save_path, wheel_crop)
                    break  # only first wheel
            else:
                plt.subplot(1, 2, 2)
                plt.text(0.5, 0.5, "No wheel detected!", fontsize=14,
                         ha='center', va='center')
                plt.axis("off")

            plt.show()