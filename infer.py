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

    ckpt_name = config["model"]["ckpt"]
    models_path = config["model"]["model_path"]

    # pull model from dvc local storage
    sp.run(["dvc", "pull", f"{models_path}/{ckpt_name}.dvc"], check=True)

    base_model = BasicBlockNet()
    model = ImageClassifier.load_from_checkpoint(
        checkpoint_path=f"{models_path}/{ckpt_name}",
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
    sp.run(["dvc", "pull", config["data"]["infer_dvc"]], check=True)

    inference_path = config["data"]["infer_data_path"]

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
    if os.path.exists(config["data"]["infer_data_path"]):
        shutil.rmtree(config["data"]["infer_data_path"])


if __name__ == "__main__":
    main()
