import base64
import io
import os

import librosa
import librosa.display
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as transforms
from flask import Flask, render_template, request
from flask_caching import Cache
from PIL import Image
from werkzeug.utils import secure_filename

from src.model import AudioModel

# Initialize the Flask app
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Load the machine learning model
model = AudioModel()
checkpoint_path = 'checkpoints/cnn/cnn_batch_size_32.pt'
if not os.path.exists(checkpoint_path):
    print('Model checkpoint not found')
else:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


# Define the route for the homepage
@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/visualize",methods = ['GET', 'POST'])
def visualize():
    return render_template("visualize.html")


# Define the image transform
image_transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

@cache.memoize()
def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate

@cache.memoize()
def generate_mel_spectrogram(waveform, sample_rate, save_file):
    # convert waveform to torch tensor
    waveform = torch.from_numpy(waveform.numpy()).float()
    transform =  T.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, hop_length=1024, n_mels=64)
    mel_spec = transform(waveform)
    mel_spec_db = T.AmplitudeToDB()(mel_spec)

    def save_and_close():
        plt.savefig(save_file)
        plt.close()

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db[0].numpy(), sr=sample_rate)
    plt.gcf().canvas.flush_events() # force update of the GUI events
    plt.gcf().canvas.get_tk_widget().after(10, save_and_close) # schedule save_and_close() in 10ms


# def generate_mel_spectrogram(waveform, sample_rate, save_file):
#     # convert waveform to torch tensor
#     waveform = torch.from_numpy(waveform.numpy()).float()
#     transform =  T.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, hop_length=1024, n_mels=64)
#     mel_spec = transform(waveform)
#     mel_spec_db = T.AmplitudeToDB()(mel_spec)

#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(mel_spec_db[0].numpy(), sr=sample_rate)
#     plt.savefig(save_file)
#     plt.close()


@app.route("/", methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join("webapp/uploaded_audio", filename))
    file_name = file.filename
    save_file = "melspectrogram.png"
    waveform, sample_rate = load_audio(os.path.join("webapp/uploaded_audio", filename))
    generate_mel_spectrogram(waveform, sample_rate,save_file)
    # Load the preprocessed image
    with open(save_file, 'rb') as f:
        image = Image.open(save_file).convert('RGB')
        input_tensor = image_transform(image)
        input_tensor = input_tensor.float()
    # Encode PNG image as base64 string
    with open(save_file, 'rb') as f:
        img_data = io.BytesIO(f.read())
        encoded_img_data = base64.b64encode(img_data.getvalue()).decode('utf-8')
    # Make prediction using the input tensor
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        predicted_idx = torch.argmax(output, dim=1)
        genre = predicted_idx.item()
        genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    return render_template("index.html",prediction = genres[genre],image_data = encoded_img_data) #,image_data = encoded_img_data


if __name__ == '__main__':
    app.run(debug=True, port=8000)
