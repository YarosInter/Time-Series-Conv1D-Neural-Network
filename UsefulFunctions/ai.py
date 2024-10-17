import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv1D, Dropout
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2


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


def run_conv1d(train_sets,
               val_sets,
               model_name,
               p_filters=32,
               p_kernel_size=5,
               p_strides=1,
               hidden_layers=2,
               p_epochs=25,
               dropout=0.20,
               lr=0.001,
               early_stopping_ptn=15,
               reduce_plateau_ptn=5,
               rp_lr=0.5,
               path="model_experiments",
               p_batch_size=256,
               kr_l2=0.1,
               verbose=0):

    """
    Builds, compiles, and trains a 1D convolutional neural network (Conv1D) model.

    Parameters:
    -----------
    train_sets : tf.data.Dataset
        Training dataset containing features and labels.
    val_sets : tf.data.Dataset
        Validation dataset for evaluating the model during training.
    model_name : str
        Name of the model.
    p_filters : int, optional
        Number of filters in the first Conv1D layer (default is 32).
    p_kernel_size : int, optional
        Size of the convolutional kernel (default is 5).
    p_strides : int, optional
        Stride length for the convolutional layers (default is 1).
    hidden_layers : int, optional
        Number of hidden Conv1D layers to add after the input layer (default is 2).
    p_epochs : int, optional
        Number of epochs to train the model (default is 25).
    dropout : float, optional
        Dropout rate applied after each Conv1D layer (default is 0.20).
    lr : float, optional
        Learning rate for the Adam optimizer (default is 0.001).
    early_stopping_ptn : int, optional
        Patience for early stopping based on validation loss (default is 15).
    reduce_plateau_ptn : int, optional
        Patience for reducing learning rate when validation loss plateaus (default is 5).
    rp_lr : float, optional
        Factor by which the learning rate is reduced when validation loss plateaus (default is 0.5).
    path : str, optional
        Directory path for saving model checkpoints (default is "model_experiments").
    p_batch_size : int, optional
        Batch size for training (default is 256).
    kr_l2 : float, optional
        L2 regularization factor for the Conv1D layers (default is 0.1).
    verbose : int, optional
        Verbosity level for training and callback outputs (default is 0).

    Returns:
    --------
    history : keras.callbacks.History
        A record of training loss and metrics at each epoch.
    """

    for features, labels in train_sets.take(1):
        features_input_shape = features.shape[1:]
        break

    model = Sequential(name=model_name)

    # Input Layer
    model.add(Conv1D(name="input_layer_0", filters=p_filters, kernel_size=p_kernel_size, strides=p_strides, padding="causal", 
                    input_shape=features_input_shape, activation="relu", kernel_regularizer=l2(kr_l2)))

    # Hidden Layers
    for i in range(0, hidden_layers):
        model.add(Conv1D(filters=int(p_filters/2), kernel_size=p_kernel_size, activation="relu", kernel_regularizer=l2(kr_l2), name=f"hidden_layer{i}"))
        model.add(Dropout(dropout, name=f"dropput_layer{i}"))

    # Applying GlobalAveragePooling1D Layer
    model.add(GlobalAveragePooling1D())
        
    # Output Layer
    model.add(Dense(1, activation="linear", name="output_layer"))

    checkpoint_callback = ai.create_model_checkpoint(model.name, path)
    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_ptn, verbose=verbose)
    reduce_plateau = ReduceLROnPlateau(monitor="val_loss", factor=rp_lr, patience=reduce_plateau_ptn, verbose=verbose)

    # Compiling the model
    model.compile(loss="mae", optimizer=Adam(learning_rate=lr), metrics=["mae"])

    # Training the model
    history = model.fit(train_sets,
                       validation_data=val_sets,
                       batch_size=p_batch_size,
                       epochs=p_epochs,
                       verbose=verbose,
                       callbacks=[checkpoint_callback, early_stopping, reduce_plateau])

    return history


