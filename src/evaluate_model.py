import numpy as np
from sklearn.metrics import accuracy_score
import torch
from models import model_general
import utils

torch.set_float32_matmul_precision("medium")

MODEL_BASE_PATH = "models_original/"
DATA_BASE_PATH = "data/"

# Choose hyperparameter setup from the lists
GRANULARITY_LIST = ["coarse", "fine"]
DATASET_SPLIT_LIST = ["Small", "Base", "Large"]
DATA_DIM_LIST = ["2D", "3D"]
MODEL_LIST = ["ResNet", "MobileNetV2", "ConvNeXt", "ViT", "Swin"]

if __name__ == "__main__":
    # Choose hypparameter setup from the lists
    granularity = GRANULARITY_LIST[0]
    dataset_split = DATASET_SPLIT_LIST[0]
    data_dim = DATA_DIM_LIST[1]
    model = MODEL_LIST[0]
    batch_size = 4
    num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the dataloaders
    data_path = f"{DATA_BASE_PATH}{dataset_split}/"
    print(f"Loading the data from: {data_path}")

    X_train, X_val, y_train, y_val = utils.get_splits(data_path)
    dataset_func = getattr(utils, f"Mozzarella{data_dim}")
    train_loader, val_loader = utils.get_data_loaders(
        dataset_func, X_train, X_val, y_train, y_val, batch_size, num_workers
    )

    # Load model from checkpoint
    model_ckpt_path = (
        f"{MODEL_BASE_PATH}{granularity}/{dataset_split}/{model}_{data_dim}.ckpt"
    )

    class model_hparams:
        data_dim = data_dim
        architecture = model
        granularity = granularity

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

    print("Accuracy:", accuracy_score(y_true, np.argmax(y_pred, 1)))
