import json
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

import config

# Agumenting the images.
working_dir = os.getcwd() + "/"
config.WORKING_DIR = working_dir

image_transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,d_type = "train",transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.d_type = d_type
        ((self.train_image_filenames,self.train_labels),(self.val_image_filenames,self.val_labels)) = self.__get_filenames_labels__()

    def __getitem__(self,index):
        if self.d_type == "train":
            image = Image.open(self.train_image_filenames[index]).convert('RGB')
            label = self.train_labels[index]
            if self.transform is not None:
                image = self.transform(image)
            return image,label
        elif self.d_type == "val":
            image = Image.open(self.val_image_filenames[index]).convert('RGB')
            label = self.val_labels[index]
            if self.transform is not None:
                image = self.transform(image)
            return image,label
        else:
            raise NotImplementedError
        
    def __len__(self):
        if self.d_type == "train":
            return len(self.train_image_filenames)
        elif self.d_type == "val":
            return len(self.val_image_filenames)
    def __get_filenames_labels__(self):
        train_image_filenames = list()
        val_image_filenames = list()
        train_labels = list()
        val_labels = list()
        class_names = sorted(os.listdir(self.data_dir))
        for class_name in class_names:
            class_dir = os.path.join(self.data_dir,class_name)
            test_size = int(len(os.listdir(class_dir))*0.2)
            for image_filename in os.listdir(class_dir)[test_size:]:
                train_image_filenames.append(os.path.join(class_dir,image_filename))
                train_labels.append(class_names.index(class_name))

            for image_filename in os.listdir(class_dir)[:test_size]:
                val_image_filenames.append(os.path.join(class_dir,image_filename))
                val_labels.append(class_names.index(class_name))
        out_file = open(config.WORKING_DIR+"dataset/"+"train_validation.json", "w")
        json.dump(
            {
            "train" : {
                "image_filenames" : train_image_filenames,
                "labels" : train_labels
            },
            "val" : {
                "image_filenames" : val_image_filenames,
                "labels" : val_labels
            }
            },out_file
        )
        return ((train_image_filenames,train_labels),(val_image_filenames,val_labels))
    
                

if __name__ == "__main__":
    train_custom_dataset = CustomDataset(config.DATA_DIR,d_type = 'train',transform=image_transform)
    val_custom_dataset = CustomDataset(config.DATA_DIR,d_type = 'val',transform=image_transform)
    total_num_samples = len(train_custom_dataset)*2
    print(f"Total number of samples before agumention: {len(train_custom_dataset)}")
    print(f"Total number of samples after agumentation: {total_num_samples}")


