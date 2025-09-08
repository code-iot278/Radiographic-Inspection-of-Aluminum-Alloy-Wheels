from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import os

def create_xray_effect(image_path, output_path):
    try:
        # ----------------------------
        # 1. Load the image
        # ----------------------------
        img = Image.open(image_path)
        original = img.copy()  # Keep original for display

        # ----------------------------
        # 2. Convert to grayscale
        # ----------------------------
        img = ImageOps.grayscale(img)

        # ----------------------------
        # 3. Adjust contrast and brightness
        # ----------------------------
        enhancer_contrast = ImageEnhance.Contrast(img)
        img = enhancer_contrast.enhance(2.0)  # Increase contrast

        enhancer_brightness = ImageEnhance.Brightness(img)
        img = enhancer_brightness.enhance(0.6)  # Decrease brightness

        # ----------------------------
        # 4. Apply blur + add random noise
        # ----------------------------
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        data = np.array(img)
        noise = np.random.normal(0, 10, data.shape)  # Mean 0, std dev 10
        data = np.clip(data + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(data)

        # ----------------------------
        # 5. Add blue-green tint
        # ----------------------------
        rgb_img = img.convert("RGB")
        overlay = Image.new("RGB", img.size, (20, 30, 40))  # Dark blue-green
        img = Image.composite(rgb_img, overlay, Image.eval(img, lambda x: 255 - x))

        # ----------------------------
        # 6. Save the image
        # ----------------------------
        img.save(output_path)

        # ----------------------------
        # 7. Display input vs output
        # ----------------------------
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.title("X-ray Effect Image")
        plt.axis("off")

        plt.show()

    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


# ----------------------------
# Process all images in main folder + subfolders
# ----------------------------
main_input_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/cropped_wheels"
output_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/xray_wheels"

for root, dirs, files in os.walk(main_input_folder):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(root, filename)

            # Maintain subfolder structure
            rel_path = os.path.relpath(root, main_input_folder)
            save_dir = os.path.join(output_folder, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            base, _ = os.path.splitext(filename)
            output_path = os.path.join(save_dir, f"{base}_xray.png")

            # Apply effect
            create_xray_effect(input_path, output_path)