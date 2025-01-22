# cifar-10-classification

## structure
'''
├── CIFAR
│   ├── lightning_logs
│   ├── model.py
│   ├── outputs
│   ├── __pycache__
│   └── trainer.py
├── conf
│   ├── config.yaml
│   ├── data
│   ├── logging
│   ├── model
│   └── training
├── data
│   ├── infer.dvc
│   ├── test.dvc
│   └── train.dvc
├── infer.py
├── lightning_logs
│   ├── version_0
│   ├── version_1
│   ├── version_2
│   ├── version_3
│   └── version_4
├── mlruns
│   ├── 0
│   ├── 648735261397303816
│   └── models
├── models
│   └── model_val_loss=1.13.ckpt
├── outputs
│   ├── 2025-01-20
│   ├── 2025-01-21
│   └── 2025-01-22
├── plots
│   ├── experiment.png
│   └── graphs.png
├── poetry.lock
├── __pycache__
│   └── calculator.cpython-311.pyc
├── pyproject.toml
├── README.md
├── test.py
└── train.py
'''
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


