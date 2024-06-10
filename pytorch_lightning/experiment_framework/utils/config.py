# Config File
# Contains Parameters for Current Run

config = {}


#==========================================================
# General Parameters
config["experiment"] = 0

#==========================================================
# System Parameters
config["system"] = {
    "accelerator": "gpu",
    "strategy": "auto",
    "num_devices": 1,
    "num_workers": 24,
}

# Data Parameters
#==========================================================
config["data"] = {
    "num_samples": 555,
    "num_sequences": 5,
    "num_features": 6,
}

# Model Paramters
#==========================================================
config["model"] = {
    "type": "LSTM",
    "num_layers": 2,
    "hidden_size": 100,
    "input_size": config["data"]["num_features"],
}

# Hyper Paramters
#==========================================================
config["hyper_parameters"] = {
    "batch_size": 10,
    "learning_rate": 0.005,
    "num_epochs": 50,
    "objective": "mse_loss", # mse_loss, mae_loss, cross_entropy
    "optimizer": "Adam", # Adam, SGD
}

# Evaluation Parameters
#==========================================================
config["evaluation"] = {
    "tags": [
        "train_error_epoch",
        "valid_error_epoch",
        # "lr-" + config["hyper_parameters"]["optimizer"]
    ],
    "show_plots": False,
    "save_plots": True,
}

# Path Parameters
#==========================================================
config["paths"] = {
    "results": "results",
    "version": 0,
}
