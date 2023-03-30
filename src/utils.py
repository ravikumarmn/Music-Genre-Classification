import json
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

import config
from model import AudioModel



def plot_losses(checkpoints):
    epochs = list(range(checkpoints["iteration"]+1))
    train_losses = checkpoints["train_losses"]
    validation_losses = checkpoints["validation_losses"]


    
    plt.plot(epochs, train_losses, label=f'Training Loss')
    plt.plot(epochs, validation_losses, label=f'Validation Loss')
    
    # Add in a title and axes labels
    plt.title(f'Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    # Display the plot
    plt.legend(loc='best')
    plt.savefig(os.path.join(config.WORKING_DIR,f"results/{config.MODEL_NAME}/{config.MODEL_NAME}_losses.png"))
    plt.clf()

def plot_accuracy(checkpoints):
    epochs = list(range(checkpoints["iteration"]+1))

    train_accuracy = checkpoints["train_accuracy"]
    validation_accuracy = checkpoints["validation_accuracy"]
    
    plt.plot(epochs, train_accuracy, label=f'Training accuracy')
    plt.plot(epochs, validation_accuracy, label=f'Validation accuracy')
    
    # Add in a title and axes labels
    plt.title(f'Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # Display the plot
    plt.legend(loc='best')
    plt.savefig(os.path.join(config.WORKING_DIR,f"results/{config.MODEL_NAME}/{config.MODEL_NAME}_accuracy.png"))
    plt.clf()
    
def predict(model, dataloader):
    y_pred = []
    y_true = []
    for inputs, labels in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())
    return y_pred, y_true

def plot_confusion_matrix(dir,val_loader):
    # Load the saved model
    model = AudioModel()
    checkpoint = torch.load(dir)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Set the model to evaluation mode
    model.eval()
    y_pred, y_true = predict(model, val_loader)

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(os.path.join(config.WORKING_DIR) +f"results/{config.MODEL_NAME}/{config.MODEL_NAME}_confustion_matrix.png")


def save_confusion(y_true,y_pred,name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(f'results/{config.MODEL_NAME}/{name}_confustion.png')

def save_model(model,name):
    # saving the model 
    import pickle 
    pickle_out = open(f"checkpoints/{config.MODEL_NAME}/{name}_classifier.pkl", mode = "wb") 
    pickle.dump(model, pickle_out) 
    pickle_out.close()


def load_images(folder_path):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            label = filename.split(".")[0]  # extract label from filename
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))  # resize image to desired size
            img_arr = np.array(img)  # convert image to numpy array
            images.append(img_arr)
            labels.append(label)
    return np.array(images), np.array(labels)

def save_analysis(json_file_str):
    result = json.load(open(json_file_str,"r"))

    values = list()
    for val in result.values():
        values.append(list(val.values()))
    vals = [list(val.keys())] + values
    model_name= [" "] + list(result.keys())
    fig = go.Figure(data=[go.Table(
    header=dict(
        values=model_name,
        fill_color='grey',
        align=['left','center'],
        font=dict(color='white', size=12)
    ),
    cells=dict(
        values=vals,
        
        line_color='darkslategray',
        align = ['left', 'center'],
        font = dict(color = 'darkslategray', size = 11)
        ))
    ])
    fig.write_image(f"results/model_performance_analysis.png")