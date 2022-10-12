import torch

NUM_CLASSES = 3
NUM_POINTS = 1024

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

# # see https://github.com/pyg-team/pytorch_geometric/issues/366
# # PLATFORM = 'windows'
# PLATFORM = 'ubuntu'  # {'windows', 'ubuntu'}
# NUM_WORKERS = 0 if PLATFORM == 'windows' else 10
NUM_WORKERS = 10

NUM_EPOCH = 32
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DECAY_RATE = 1e-4
LEARNING_RATE_CLIP = 1e-5
LEARNING_RATE_DECAY = 0.7
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECCAY = 0.5
MOMENTUM_DECCAY_STEP = 10