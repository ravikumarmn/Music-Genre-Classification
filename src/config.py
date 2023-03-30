WORKING_DIR = "/home/ravikumar/Developer/computer_vision/music_genre/"
DATA_DIR = WORKING_DIR + "dataset/images_original/"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 100
DEVICE = 'cpu'
PATIENCE = 5
MODEL_NAME = 'cnn'#random_forest #cnn #svm
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
TEST_SIZE = 0.2
RESULT_FILE = F"results/performance_analysis.json"

