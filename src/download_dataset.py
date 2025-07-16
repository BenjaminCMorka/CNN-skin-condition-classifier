import kagglehub
import os
import shutil


download_path = kagglehub.dataset_download("shubhamgoel27/dermnet")


base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
train_dir = os.path.join(base_data_dir, 'train')
test_dir = os.path.join(base_data_dir, 'test')


os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


source_train_dir = os.path.join(download_path, 'train')
source_test_dir = os.path.join(download_path, 'test')

# only keep acne eczema
allowed_classes = {'acne and rosacea photos', 'eczema photos'}

def copy_selected_classes(src_dir, dest_dir):
    for class_name in os.listdir(src_dir):
        if class_name.lower() in allowed_classes:
            src_class_path = os.path.join(src_dir, class_name)
            dest_class_path = os.path.join(dest_dir, class_name)
            print(f"Copying {class_name} to {dest_class_path}")
            shutil.copytree(src_class_path, dest_class_path, dirs_exist_ok=True)


copy_selected_classes(source_train_dir, train_dir)
copy_selected_classes(source_test_dir, test_dir)

print(f"Finished preparing data in {base_data_dir}")
