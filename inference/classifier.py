import torch

import config
from src.model import AudioModel

CHECKPOINT_STR = "cnn_batch_size_32.pt"
CHECKPOINT_DIR = f"{config.WORKING_DIR}checkpoints/{config.MODEL_NAME}/{CHECKPOINT_STR}"

model = AudioModel()

checkpoint = torch.load(CHECKPOINT_DIR)
model.load_state_dict(checkpoint["model_state_dict"])

print(checkpoint.keys())
print()