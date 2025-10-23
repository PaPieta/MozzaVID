# Minimal example of training a model on the MozzaVID dataset (using the lightning framework)

import datetime

import torch
import lightning as L

import utils_local, utils_stream
from models import model_general

torch.set_float32_matmul_precision("medium")

MODELS_BASE_PATH = "models_new/"
DATA_BASE_PATH = "data/DTU/" # Set if data is stored locally, streaming path is hard coded in utils_stream.py
DATA_MODE = "stream"  # "local" or "stream"

# Choose hyperparameter setup from the lists
GRANULARITY_LIST = ["coarse", "fine"]
DATASET_SPLIT_LIST = ["Small", "Base", "Large"]
DATA_DIM_LIST = ["2D", "3D"]
MODEL_TYPE_LIST = ["ResNet", "MobileNetV2", "ConvNeXt", "ViT", "Swin"]

GRANULARITY = GRANULARITY_LIST[0]
DATASET_SPLIT = DATASET_SPLIT_LIST[0]
DATA_DIM = DATA_DIM_LIST[0]
MODEL_TYPE = MODEL_TYPE_LIST[0]

LR = 0.001  # Learning rate
NUM_EPOCHS = 10
BATCH_SIZE = 4
NUM_WORKERS = 4
ROTATE = False  # Additional rotation of the data, used in the ablation study

def train():
    timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")

    # Prepare the dataloaders
    if DATA_MODE == "local":
        data_path = f"{DATA_BASE_PATH}{DATASET_SPLIT}/"

        X_train, X_val, X_test, y_train, y_val, y_test = utils_local.get_splits(data_path, GRANULARITY)
        dataset_func = getattr(utils_local, f"Mozzarella{DATA_DIM}")
        train_loader, val_loader, test_loader = utils_local.get_data_loaders(
            dataset_func, X_train, X_val, X_test, y_train, y_val, y_test,
            BATCH_SIZE,
            NUM_WORKERS,
            rotate=ROTATE,
        )
    elif DATA_MODE == "stream":
        data_path = f"HuggingFace: {DATASET_SPLIT}"
        print(f"Streaming the {DATASET_SPLIT} data split from HuggingFace")
        train_loader, val_loader, test_loader = utils_stream.get_data_loaders(
            DATASET_SPLIT, data_dim=DATA_DIM, granularity=GRANULARITY,
            rotate=ROTATE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    else:
        raise ValueError("DATA_MODE must be 'local' or 'stream'")
    
    # Setup model
    class model_hparams:
        data_dim = DATA_DIM
        architecture = MODEL_TYPE
        granularity = GRANULARITY

    model_name = f"{MODEL_TYPE}_{DATA_DIM}_{timestamp}"
    if ROTATE:
        model_name = f"rotated_{model_name}"

    # Load model
    model = model_general.Net(model_hparams=model_hparams, lr=LR)

    # Prepare training callbacks
    callbacks = []

    model_path = f"{MODELS_BASE_PATH}/{GRANULARITY}/{DATASET_SPLIT}/{model_hparams.data_dim}/{model_name}"
    # Save best model callback
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=model_path, filename=model_name, monitor="val/loss"
    )
    callbacks.append(checkpoint_callback)
    # # Add early stopping
    # early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=30, verbose=False, mode="min")
    # callbacks.append(early_stop_callback)

    print(f"Model name: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Dataset: {data_path}")
    print(f"Cuda available: {torch.cuda.is_available()}")

    # Train model
    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        precision="16-mixed",
        log_every_n_steps=1,
        callbacks=callbacks,
        accelerator="auto",
        accumulate_grad_batches=1,
    )
    trainer.fit(model, train_loader, val_loader)

    # Test model
    trainer.test(model, test_loader, ckpt_path="best")

if __name__ == "__main__":
    train()
