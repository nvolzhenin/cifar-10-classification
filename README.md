# cifar-10-classification

## Project structure
.
(0tqq (BCIFAR
(0x   tqq (Blightning_logs
(0x   x   mqq (Bversion_0
(0x   tqq (Boutputs
(0x   x   mqq (B2025-01-21
(0x   tqq (B__pycache__
(0x   tqq (Bmodel.py
(0x   mqq (Btrainer.py
(0tqq (Bconf
(0x   tqq (Bdata
(0x   x   mqq (Bdata.yaml
(0x   tqq (Blogging
(0x   x   mqq (Blogging.yaml
(0x   tqq (Bmodel
(0x   x   mqq (Bmodel.yaml
(0x   tqq (Btraining
(0x   x   mqq (Btraining.yaml
(0x   mqq (Bconfig.yaml
(0tqq (Bdata
(0tqq (Blightning_logs
(0x   tqq (Bversion_0
(0x   x   tqq (Bhparams.yaml
(0x   x   mqq (Bmetrics.csv
(0x   tqq (Bversion_1
(0x   x   tqq (Bhparams.yaml
(0x   x   mqq (Bmetrics.csv
(0x   tqq (Bversion_2
(0x   x   tqq (Bhparams.yaml
(0x   x   mqq (Bmetrics.csv
(0x   tqq (Bversion_3
(0x   x   tqq (Bhparams.yaml
(0x   x   mqq (Bmetrics.csv
(0x   mqq (Bversion_4
(0x       tqq (Bhparams.yaml
(0x       mqq (Bmetrics.csv
(0tqq (Bmlruns
(0x   tqq (B0
(0x   x   mqq (Bmeta.yaml
(0x   tqq (B648735261397303816
(0x   x   tqq (B10b8137f241345188faf8ea002c0976f
(0x   x   tqq (B17f6decc5959498da9988373e6351a6c
(0x   x   tqq (B3c9ef9b939804e0ab1bca938ef404c04
(0x   x   tqq (Ba9e67558d03444e2b9b0b29a114fa143
(0x   x   tqq (Bc42e7c51328e42e291c654277ea4c52d
(0x   x   mqq (Bmeta.yaml
(0x   mqq (Bmodels
(0tqq (Bmodels
(0x   mqq (Bmodel_val_loss=1.13.ckpt
(0tqq (Boutputs
(0x   tqq (B2025-01-20
(0x   x   tqq (B20-20-50
(0x   x   tqq (B20-21-40
(0x   x   tqq (B20-21-53
(0x   x   tqq (B20-22-18
(0x   x   mqq (B20-24-03
(0x   tqq (B2025-01-21
(0x   x   tqq (B12-49-54
(0x   x   tqq (B12-50-40
(0x   x   tqq (B12-51-26
(0x   x   tqq (B12-54-00
(0x   x   tqq (B12-56-30
(0x   x   tqq (B14-00-34
(0x   x   tqq (B14-02-34
(0x   x   tqq (B15-59-59
(0x   x   tqq (B16-00-25
(0x   x   tqq (B16-29-03
(0x   x   tqq (B16-30-13
(0x   x   tqq (B19-43-44
(0x   x   tqq (B19-48-19
(0x   x   tqq (B19-48-53
(0x   x   mqq (B19-51-21
(0x   mqq (B2025-01-22
(0x       tqq (B13-04-32
(0x       tqq (B13-11-50
(0x       tqq (B13-14-01
(0x       tqq (B13-52-04
(0x       tqq (B16-25-13
(0x       tqq (B16-25-37
(0x       tqq (B16-48-28
(0x       mqq (B16-49-40
(0tqq (Bplots
(0x   tqq (Bexperiment.png
(0x   mqq (Bgraphs.png
(0tqq (B__pycache__
(0tqq (Binfer.py
(0tqq (Bpoetry.lock
(0tqq (Bpyproject.toml
(0tqq (BREADME.md
(0tqq (Btest.py
(0mqq (Btrain.py

63 directories, 28 files



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


