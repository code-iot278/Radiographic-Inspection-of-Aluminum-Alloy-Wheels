import os
import shutil

# Input dataset folders
folder1 = "/content/drive/MyDrive/Colab Notebooks/Wheel/xray_wheels"
folder2 = "/content/drive/MyDrive/Colab Notebooks/Wheel/collected Dataset"

# Output merged folder
output_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/merged_dataset"
os.makedirs(output_folder, exist_ok=True)

# Copy all images from both folders
for folder in [folder1, folder2]:
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                src_path = os.path.join(root, filename)

                # Ensure unique filenames
                rel_path = os.path.relpath(root, folder)
                save_dir = os.path.join(output_folder, rel_path)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, filename)
                if os.path.exists(save_path):
                    name, ext = os.path.splitext(filename)
                    save_path = os.path.join(save_dir, f"{name}_copy{ext}")

                shutil.copy2(src_path, save_path)

print("✅ Dataset folders merged successfully!")