import os
import shutil

# Define paths
source_folder = 'oldata'
images_folder = os.path.join( 'data/images')
labels_folder = os.path.join( 'data/labels')

# Create destination folders if they don't exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)
qa
# Get list of all files in the source folder
all_files = os.listdir(source_folder)

# Build a set of all label basenames (without extension)
label_basenames = {os.path.splitext(f)[0] for f in all_files if f.endswith('.txt')}

# Process image files
for file in all_files:
    if file.endswith(('.jpg', '.png')):
        base_name, ext = os.path.splitext(file)
        label_file = base_name + '.txt'

        if base_name in label_basenames:
            # Move image
            shutil.move(os.path.join(source_folder, file), os.path.join(images_folder, file))
            # Move corresponding label
            if os.path.exists(os.path.join(source_folder, label_file)):
                shutil.move(os.path.join(source_folder, label_file), os.path.join(labels_folder, label_file))

print("Files with matching labels have been moved.")
