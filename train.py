import os
import shutil
import subprocess as sp

import hydra
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from CIFAR.model import BasicBlockNet
from CIFAR.trainer import ImageClassifier


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig) -> None:
    transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                config["model"]["image_mean"], config["model"]["image_std"]
            ),
        ]
    )

    # pull data from dvc local storage
    sp.run(
        ["dvc", "pull", config["data"]["train_dvc"]],
        check=True,
    )

    # create dataset and divide it into train and validation parts
    trainval = torchvision.datasets.ImageFolder(
        os.path.join(config["data"]["train_data_path"]), transform=transformer
    )

    train_size = int((1 - config["training"]["val_size"]) * len(trainval))
    val_size = len(trainval) - train_size

    train_dataset, val_dataset = random_split(
        trainval,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    model = BasicBlockNet()
    module = ImageClassifier(model, lr=config["training"]["lr"])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=config["model"]["model_path"],
        filename="model_{val_loss:.2f}",
        save_top_k=config["training"]["save_top_k"],
        mode="min",
    )

    logger = pl.loggers.MLFlowLogger(
        experiment_name=config["logging"]["experiment_name"],
        run_name=config["logging"]["run_name"],
        tracking_uri=config["logging"]["tracking_uri"],
    )

    logger.log_hyperparams(
        {
            "learning_rate": config["training"]["lr"],
            "batch_size": config["training"]["batch_size"],
            "n_epochs": config["training"]["n_epochs"],
        }
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["n_epochs"],
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(module, train_loader, val_loader)

    # delete folder with train dataset
    if os.path.exists(config["data"]["train_data_path"]):
        shutil.rmtree(config["data"]["train_data_path"])


if __name__ == "__main__":
    main()
