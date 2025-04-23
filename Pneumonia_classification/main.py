import torch
import torch.nn as nn
from src.model import Model
from src.utils import load_dataset

if __name__ == "__main__":
    opt = {
    # Model
    "model": "CNN",
    "iter": 0,

    # Networks
    "networks": [
        {
            "name": "CNN",
            "type": "CNN",
            "args": {            },
            "path": "./pretrained_models/CNN.pth"
        },
        
    ],

    # Datasets
    "train_datasets": [
        {
            "type": "x_ray_TrainDataset",
            "path": 'data/train/',
            "shuffle": True,
            "args": {},
            "batch_size": 8
        }
    ],
    "test_datasets": [
        {
            "type": "x_ray_TestDataset",
            "path": 'data/test/',
            "shuffle": False,
            "args": {}
        }
    ],

    # Training parameters
    "lr": 3e-4,
    "batch_size": 8,
    "momentum": 0.9,
    "num_folds": 5,
    "print_every": 1,
    "save_every": 10,
    "validate_every": 10,
    "num_iters": 120,
    "log_dir": "pretrained_models",
    "log_file": "logs/log.out"

    }

    model = Model()
    train_images, train_labels = load_dataset(opt["train_datasets"][0])
    test_images, test_labels = load_dataset(opt["test_datasets"][0])
    print('Datasets ready')

    model.train(train_images, train_labels)
    
