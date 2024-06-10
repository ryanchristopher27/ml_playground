import torch
from torch.utils.data import TensorDataset
import pandas as pd
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import os
from sklearn.preprocessing import MinMaxScaler
import yaml

from utils.models import Network
from utils.data import create_stock_dataset, StockDataModule
from utils.helpers import *
from utils.plots import *
# from utils.config import config



def stock_experiment(config: dict, plot_results: bool):

    # Load Config Variables
    task = config["experiment"]
    path_results = config["paths"]["results"]
    num_epochs = config["hyper_parameters"]["num_epochs"]
    batch_size = config["hyper_parameters"]["batch_size"]
    strategy = config["system"]["strategy"]
    accelerator = config["system"]["accelerator"]
    num_devices = config["system"]["num_devices"]
    num_features = config["data"]["num_features"]
    seq_length = config["data"]["num_sequences"]

    show_plots = config["evaluation"]["show_plots"]
    save_plots = config["evaluation"]["save_plots"]

    train_scaler = MinMaxScaler(feature_range=(0,1))
    test_scaler = MinMaxScaler(feature_range=(0,1))


    # Create Train and Dest Datasets
    X_train, y_train, X_test, y_test, train_scaler, test_scaler = create_stock_dataset(num_features=num_features, seq_len=seq_length, train_scaler=train_scaler, test_scaler=test_scaler)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    # print(X_train.shape)

    config["data"]["num_samples"] = X_train.shape[0]

    # Data Module
    data_module = StockDataModule(batch_size, train_dataset, test_dataset, test_dataset)
    # data_module.setup()

    # Create Model
    model = Network(config)

    # Create: Logger
    path_save = path_results + "/exp_%s" % task
    exp_logger = CSVLogger(save_dir=path_save)

    # Create: Trainer
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(callbacks=[lr_monitor], 
                        logger=exp_logger, devices=num_devices, 
                        log_every_n_steps=1, max_epochs=num_epochs, 
                        strategy=strategy, accelerator=accelerator)

    # Train: Model

    trainer.fit(model=model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

    target_names = config["evaluation"]["tags"]

    # Gather: Most Recent Trained Model (Highest Version)
    path_version = os.path.join(path_save, "lightning_logs")
    version = get_latest_version(path_version)

    # Open the file in write mode
    with open(os.path.join(path_version, "version_%s" % version, 'config.yaml'), 'w') as file:
        # Use yaml.dump() to write the dictionary to the file
        yaml.dump(config, file)

    # Update: Paths
    path_file = os.path.join(path_version, "version_%s" % version, "metrics.csv")

    if plot_results:
        path_plots = os.path.join(path_version, "version_%s" % version, "plots")

        os.makedirs(path_plots, exist_ok=True)

        # Visualize: Training Analytics

        get_training_results(path_file, path_plots, target_names, show_plots, save_plots)

    test_stats = trainer.test(model=model, dataloaders=data_module.test_dataloader())
    trainer.predict(model=model, dataloaders=data_module.test_dataloader())

    test_error_epoch = test_stats[0]["test_error_epoch"]
    # print(test_stats)

    predictions = np.concatenate(model.test_predictions)

    dummy_features_pred = np.zeros((predictions.shape[0], 6))
    predictions = predictions.reshape(-1)
    dummy_features_pred[:, 0] = predictions
    unnormalized_predictions = test_scaler.inverse_transform(dummy_features_pred)[:, 0]

    df_predictions = pd.DataFrame(unnormalized_predictions, columns=["Predicted"])

    dummy_features_y_test = np.zeros((y_test.shape[0], 6))
    y_test = y_test.reshape(-1)
    dummy_features_y_test[:, 0] = y_test
    unnormalized_y_test = test_scaler.inverse_transform(dummy_features_y_test)[:, 0]

    df_actual = pd.DataFrame(unnormalized_y_test, columns=["Actual"])

    # Ensure that the number of predictions matches the number of actual values
    if len(df_predictions) > len(df_actual):
        df_predictions = df_predictions[:len(df_actual)]
    elif len(df_actual) > len(df_predictions):
        df_actual = df_actual[:len(df_predictions)]

    # Create a new DataFrame by concatenating the predicted and actual values
    df_combined = pd.concat([df_predictions, df_actual], axis=1)

    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)

    dummy_features_y_train = np.zeros((y_train.shape[0], 6))
    y_train = y_train.reshape(-1)
    dummy_features_y_train[:, 0] = y_train
    unnormalized_y_train = train_scaler.inverse_transform(dummy_features_y_train)[:, 0]

    df_train = pd.DataFrame(unnormalized_y_train, columns=["Train"])

    if plot_results:

        # Concatenate the train data with the combined DataFrame
        df_plot = pd.concat([df_train, df_combined], ignore_index=True)

        # '''
        # Plot the train data, predicted values, and actual values
        plt.figure(figsize=(10, 6))
        plt.plot(df_plot.index[:len(df_train)], df_plot['Train'][:len(df_train)], label='Train')
        plt.plot(df_plot.index[len(df_train):], df_plot['Predicted'][len(df_train):], label='Predicted')
        plt.plot(df_plot.index[len(df_train):], df_plot['Actual'][len(df_train):], label='Actual')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Train, Predicted, and Actual Values')
        plt.legend()
        plt.grid(True)

        if save_plots:
            path = path_plots + f"/train_and_test.png"
            plt.savefig(path)
            plt.close()

        if show_plots:
            plt.show()

        # plt.show()
        # '''

        # '''
        # Plot the predicted and actual values
        plt.figure(figsize=(10, 6))
        plt.plot(df_combined.index, df_combined['Predicted'], label='Predicted')
        plt.plot(df_combined.index, df_combined['Actual'], label='Actual')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        plt.grid(True)

        if save_plots:
            path = path_plots + f"/test.png"
            plt.savefig(path)
            plt.close()

        if show_plots:
            plt.show()

        # plt.show()
        # '''

    return df_predictions, df_actual, test_error_epoch