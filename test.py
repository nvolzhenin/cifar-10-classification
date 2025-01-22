import os
import shutil
import subprocess as sp

import hydra
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader

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
    sp.run(["dvc", "pull", config["data"]["test_dvc"]], check=True)

    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(config["data"]["train_data_path"]), transform=transformer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    model = BasicBlockNet()
    module = ImageClassifier.load_from_checkpoint(
        f"{config['model']['model_path']}/{config['model']['ckpt']}",
        model=model,
        lr=config["training"]["lr"],
    )

    trainer = pl.Trainer(accelerator="auto", devices="auto")

    results = trainer.test(module, dataloaders=test_loader)
    print(results)

    # delete folder with train dataset
    if os.path.exists(config["data"]["test_data_path"]):
        shutil.rmtree(config["data"]["test_data_path"])


if __name__ == "__main__":
    main()
