import os
import shutil
import subprocess as sp

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from CIFAR.model import BasicBlockNet
from CIFAR.trainer import ImageClassifier


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig) -> None:

    base_model = BasicBlockNet()
    model = ImageClassifier.load_from_checkpoint(
        ckpt_path=f"{config['model']['model_path']}/{config['model']['ckpt']}",
        model=base_model,
        lr=config["training"]["lr"],
    )
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                config["model"]["image_mean"], config["model"]["image_std"]
            ),
        ]
    )

    # pull data from dvc local storage
    sp.run(["dvc", "pull", config["data_loading"]["infer_dvc"]], check=True)

    inference_path = config["data_loading"]["infer_data_path"]

    for filename in os.listdir(inference_path):
        img_path = os.path.join(inference_path, filename)
        if img_path.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0)

            with torch.no_grad():
                output = model(image)

            probabilities = torch.nn.functional.softmax(output, dim=1)

            class_num = torch.argmax(probabilities, dim=1).item()
            class_text = config["model"]["class_dict"][class_num]

            print(f"Image: {filename},Class: {class_text}")

    # delete folder with train dataset
    if os.path.exists(config["data_loading"]["infer_data_path"]):
        shutil.rmtree(config["data_loading"]["infer_data_path"])


if __name__ == "__main__":
    main()
