import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from skimage import io
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.optim import SGD

import config
from dataset import CustomDataset, image_transform
from model import AudioModel
from utils import (plot_confusion_matrix, plot_losses, save_analysis,
                   save_confusion, save_model,plot_accuracy)

device = config.DEVICE

print(f"Using Device : {device}")
working_dir = os.getcwd() + "/"
config.WORKING_DIR = working_dir

train_custom_dataset = CustomDataset(config.DATA_DIR,d_type = 'train',transform=image_transform)
val_custom_dataset = CustomDataset(config.DATA_DIR,d_type = 'val',transform=image_transform)
train_loader = data.DataLoader(train_custom_dataset, batch_size=config.BATCH_SIZE, shuffle=True,)
val_loader = data.DataLoader(val_custom_dataset, batch_size=config.BATCH_SIZE, shuffle=False)



model = AudioModel()
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
train_losses = []
val_losses = []


def main():
    validation_loss = np.inf
    train_accuracy = list()
    validation_accuracy = list()
    for epoch in range(config.EPOCHS):
        # Training loop
        model.train()
        train_total_loss = 0
        train_total = 0
        train_correct = 0
        for inputs, labels in train_loader:
            
            optimizer.zero_grad()
            outputs = model(inputs) # inputs : torch.Size([8, 3, 224, 224]
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_total_loss += loss.item()
            train_correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

        train_acc = 100 * train_correct / train_total
        train_loss = train_total_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)

        # Validation loop
        # validation_accuracy = list()

        with torch.no_grad():
            model.eval()
            val_total_loss = 0
            y_pred = []
            y_true = []
            # val_total = 0
            # val_correct = 0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # val_total += labels.size(0)
                train_total_loss += loss.item()
                # val_correct += (predicted == labels).sum().item()


                y_pred.extend(predicted.tolist())
                y_true.extend(labels.tolist())
            val_loss = val_total_loss / len(val_loader)
            val_losses.append(val_loss)
            accuracy = accuracy_score(y_true, y_pred)*100  # calculate accuracy
            validation_accuracy.append(accuracy)
            get_result(y_true,y_pred,"cnn",config.RESULT_FILE)

        directory = "checkpoints/" + config.MODEL_NAME
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(f"Epoch {epoch+1}/{config.EPOCHS} >>>>> train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={accuracy:.2f}%")

        if val_loss < validation_loss:
            validation_loss = val_loss
            early_stopping = 0
            torch.save(
                {
                    "iteration" : epoch,
                    "model_state_dict" : model.state_dict(),
                    "train_losses" : train_losses,
                    "validation_losses": val_losses,
                    "train_accuracy" : train_accuracy,
                    "validation_accuracy" : validation_accuracy,
                    "params" : {k:v for k,v in config.__dict__.items() if "__" not in k},
                    "learning_rate":config.LEARNING_RATE
                },
                os.path.join(directory,f'{config.MODEL_NAME}_batch_size_{config.BATCH_SIZE}.pt')
            )
        else:
            early_stopping += 1
        if early_stopping == config.PATIENCE:
            print(f"EARLY STOPPING AT EPOCH {epoch+1}")
            break


# Load training images
def get_images(X_train_filenames,X_test_filenames):
    X_train = []
    for filename in X_train_filenames:
        img = Image.open(filename)
        # img = img.resize((64, 64))  # Resize the image to 64x64
        img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
        img_data = img_data.flatten()
        X_train.append(img_data)
    X_train = np.array(X_train)

    # Load and reshape the testing images
    X_test = []
    for filename in X_test_filenames:
        img = Image.open(filename)
        img = img.resize((64, 64))  # Resize the image to 64x64
        img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
        img_data = img_data.flatten()
        X_test.append(img_data)
    X_test = np.array(X_test)

    return X_train,X_test


def edit_json(file,data_dict):
    json_file = json.load(open(file,"r")) # a
    json_file.update(data_dict) # update
    json.dump(json_file, open(file,"w")) # 

def get_result(y_test,y_pred,name,save_file):
    analysis = dict(   
        accuracy = round(accuracy_score(y_test, y_pred),3),
        precision = round(precision_score(y_test, y_pred,average='weighted'),3) ,
        recall = round(recall_score(y_test, y_pred,average='weighted'),3),
        f_score = round(f1_score(y_test, y_pred,average='weighted'),3)
        )
    result = {name.upper() : analysis}
    edit_json(save_file,result)
    print(f"\nAccuracy of model {name} : {analysis['accuracy']:.3f}")
        

def train_model(model,X_train, y_train,X_test,y_true):
    if model == "svm":
        print("Training...!")
        from sklearn.svm import SVC

        svc = SVC(kernel='linear',gamma='auto')
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        save_model(svc,model)
        save_confusion(y_true,y_pred,model)
        get_result(y_true,y_pred,model,config.RESULT_FILE)
        print(f"Accuracy on validation data for {model} is ",accuracy_score(y_true,y_pred))
    elif model == "random_forest":
        print("Training...!")
        from sklearn.ensemble import RandomForestClassifier
        rfc =RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state = 42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
        save_model(rfc,model)
        save_confusion(y_true,y_pred,model)
        get_result(y_true,y_pred,model,config.RESULT_FILE)

        print(f"Accuracy on validation data for {model} is ",accuracy_score(y_true,y_pred))
    else:
        raise NotImplementedError

if __name__ == "__main__":
   print(f"Using {config.MODEL_NAME} model\n")
   directory = config.WORKING_DIR + "checkpoints/" + config.MODEL_NAME
   dir = os.path.join(directory,f'{config.MODEL_NAME}_batch_size_{config.BATCH_SIZE}.pt')
   dirs = f"{config.WORKING_DIR}results/{config.MODEL_NAME}"

   if not os.path.exists(dirs):
        try:
           os.mkdir(dirs)
        except OSError as error:
            print("Directory '%s' can not be created")
    
   if not os.path.exists(directory):
        try:
           os.mkdir(directory)
        except OSError as error:
            print("Directory '%s' can not be created")
   if config.MODEL_NAME != "cnn":
        data_dir = config.DATA_DIR
        genres = config.GENRES

        images = []
        labels = []
        for genre in genres:
            genre_dir = os.path.join(data_dir, genre)
            for image_file in os.listdir(genre_dir):
                image_path = os.path.join(genre_dir, image_file)
                image = Image.open(image_path).convert('L')  # Convert to grayscale
                image = image.resize((64, 64))  # Resize image
                image = np.array(image).flatten()  # Flatten image
                images.append(image)
                labels.append(genres.index(genre))
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=config.TEST_SIZE, random_state=42)

        train_model(config.MODEL_NAME,X_train, y_train,X_test,y_test)
   elif config.MODEL_NAME == "cnn":
        total_num_samples = len(train_custom_dataset)*2
        
        print(f"Total number of samples before agumention: {len(train_custom_dataset)}")
        print(f"Total number of samples after agumentation: {total_num_samples}\n")
        main()
        print(f"Traning finished.\nCheckpoints saved to {dir}")
        checkpoints = torch.load(dir)
        plot_losses(checkpoints)
        plot_accuracy(checkpoints)
        plot_confusion_matrix(dir,val_loader)

   else:
       raise NotImplementedError
   save_analysis("results/performance_analysis.json")



