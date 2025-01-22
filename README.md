# cifar-10-classification

## Project structure
├── CIFAR                               
│   ├── model.py
│   └── trainer.py
├── conf
│   ├── config.yaml
│   ├── data
│   │   └── data.yaml
│   ├── logging
│   │   └── logging.yaml
│   ├── model
│   │   └── model.yaml
│   └── training
│       └── training.yaml
├── data
│   ├── infer.dvc
│   ├── test.dvc
│   └── train.dvc
├── infer.py
├── models
│   └── model_val_loss=1.13.ckpt
├── plots
│   ├── experiment.png
│   └── graphs.png
├── poetry.lock
├── pyproject.toml
├── README.md
├── test.py
└── train.py

Тема проекта - многоклассовая классификация изображений с помощью нейронных сетей

Датасет - планируется взять CIFAR10 - https://www.cs.toronto.edu/~kriz/cifar.html - (не 100% именно его, а если и его, то, возможно, не весь)

Почему такой датасет - картинки в нём небольшие, датасет уже разбит на train/test


Решение - с помощью свёрточных нейронных сетей

Требуемые библиотеки - 

torch,
torch.nn,
torch.nn.functional,
torch.optim,
torchvision,
torchvision.transforms,


Этапы подготовки:

1) Сбор данных (в данном случае можно просто загрузить через torch.Datasets)

2) Подготовка данных (разбиение на train/test, выделение валидационной выборки)

3) Обучение модели, подбор гиперпараметров

4) Оценка модели

5) Деплой модели - контейнеризация в Docker

6) Настройка мониторинга - MLFlow


Предполагаемый сервис - классификация изображения - тг-бот или сервис на FastAPI


