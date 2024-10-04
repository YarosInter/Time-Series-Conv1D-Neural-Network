import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv1D, Dropout
from tensorflow.keras.optimizers import Adam
import os


def create_model_checkpoint(model_name, save_path="model_experiments"):
    """
    Creates a ModelCheckpoint callback to save the best-performing version of a model during training.
    
    Args:
        model_name (str): The name of the model to be used for saving the file.
        save_path (str, optional): The directory path where the model file will be saved. Defaults to "model_experiments".
    
    Returns:
        ModelCheckpoint: A callback that saves the model with the lowest validation loss.
    
    Notes:
        - The checkpoint saves the model in the provided directory (`save_path`) with the format "{model_name}.keras".
        - Only the model with the best validation loss is saved.
    """
    # Ensure the directory exists
    #os.makedirs(save_path, exist_ok=True)
    
    return ModelCheckpoint(
        filepath=os.path.join(save_path, f"{model_name}.keras"),
        monitor="val_loss",
        verbose=0,
        save_best_only=True
    )