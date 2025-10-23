import numpy as np
from sklearn.metrics import accuracy_score
import torch
from models import model_general
import utils_local, utils_stream

torch.set_float32_matmul_precision("medium")

MODEL_BASE_PATH = "models/"
DATA_BASE_PATH = "data/DTU/" # Set if data is stored locally, streaming path is hard coded in utils_stream.py
DATA_MODE = "stream"  # "local" or "stream"

# Choose hyperparameter setup from the lists
GRANULARITY_LIST = ["coarse", "fine"]
DATASET_SPLIT_LIST = ["Small", "Base", "Large"]
DATA_DIM_LIST = ["2D", "3D"]
MODEL_LIST = ["ResNet", "MobileNetV2", "ConvNeXt", "ViT", "Swin"]

GRANULARITY = GRANULARITY_LIST[0]
DATASET_SPLIT = DATASET_SPLIT_LIST[0]
DATA_DIM = DATA_DIM_LIST[0]
MODEL = MODEL_LIST[0]

BATCH_SIZE = 4
NUM_WORKERS = 0

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the dataloaders
    if DATA_MODE == "local":
        data_path = f"{DATA_BASE_PATH}{DATASET_SPLIT}/"
        print(f"Loading the data from: {data_path}")

        X_train, X_val, X_test, y_train, y_val, y_test = utils_local.get_splits(data_path, GRANULARITY)
        dataset_func = getattr(utils_local, f"Mozzarella{DATA_DIM}")
        train_loader, val_loader, test_loader = utils_local.get_data_loaders(
            dataset_func, X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE, NUM_WORKERS
        )
    elif DATA_MODE == "stream":
        print(f"Streaming the {DATASET_SPLIT} data split from HuggingFace")
        train_loader, val_loader, test_loader = utils_stream.get_data_loaders(
            DATASET_SPLIT, data_dim=DATA_DIM, granularity=GRANULARITY,
            rotate=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    else:
        raise ValueError("DATA_MODE must be 'local' or 'stream'")

    # Load model from checkpoint
    model_ckpt_path = (
        f"{MODEL_BASE_PATH}{GRANULARITY}/{DATASET_SPLIT}/{MODEL}_{DATA_DIM}.ckpt"
    )

    class model_hparams:
        data_dim = DATA_DIM
        architecture = MODEL
        granularity = GRANULARITY

    print(f"Loading the model from: {model_ckpt_path}")
    model = model_general.Net.load_from_checkpoint(
        model_hparams=model_hparams, checkpoint_path=model_ckpt_path
    )
    model.eval()

    # Make predictions using validation data
    y_pred = []
    y_true = []
    for batch in iter(val_loader):
        x_in = batch[0].type(torch.FloatTensor).to(device)

        y_out = model(x_in)

        y_pred.append(y_out.cpu().detach().numpy())
        y_true.append(batch[1].cpu().detach().numpy())

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    print("Validation accuracy:", accuracy_score(y_true, np.argmax(y_pred, 1)))

    # Make predictions using test data
    y_pred = []
    y_true = []
    for batch in iter(test_loader):
        x_in = batch[0].type(torch.FloatTensor).to(device)

        y_out = model(x_in)

        y_pred.append(y_out.cpu().detach().numpy())
        y_true.append(batch[1].cpu().detach().numpy())
    
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    print("Test accuracy:", accuracy_score(y_true, np.argmax(y_pred, 1)))
