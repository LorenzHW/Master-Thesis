import numpy as np


def main():
    temp = np.load("./values.npy", allow_pickle=True)
    temp = temp.item()

    transferred_data = {
        "train_epoch_100_y_vs_pred": temp.get("train_epoch_100_y_vs_pred"),
        "train_epoch_100_external_model_loss_binary": temp.get("train_epoch_100_external_model_loss_binary"),
        "train_epoch_100_external_model_loss_raw": temp.get("train_epoch_100_external_model_loss_raw"),
        "train_epoch_100_z": temp.get("train_epoch_100_z"),
        "train_epoch_100_label": temp.get("train_epoch_100_label"),
        "validation_epoch_100_y_vs_pred": temp.get("validation_epoch_100_y_vs_pred"),
        "validation_epoch_100_external_model_loss_binary": temp.get("validation_epoch_100_external_model_loss_binary"),
        "validation_epoch_100_external_model_loss_raw": temp.get("validation_epoch_100_external_model_loss_raw"),
        "validation_epoch_100_z": temp.get("validation_epoch_100_z"),
        "validation_epoch_100_label": temp.get("validation_epoch_100_label"),
        "test_data_y_vs_pred": temp.get("test_data_y_vs_pred"),
        "test_data_external_model_loss_binary": temp.get("test_data_external_model_loss_binary"),
        "test_data_external_model_loss_raw": temp.get("test_data_external_model_loss_raw"),
        "test_data_z": temp.get("test_data_z"),
        "test_data_label": temp.get("test_data_label"),
    }
    np.save("./temp.npy", transferred_data)
    print("HELLO WORLD")


if __name__ == '__main__':
    main()
