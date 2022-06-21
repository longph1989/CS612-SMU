# CS612-SMU

## Installation

First install virtual enviroment and activate it.

```
virtualenv -p python3 cs612_venv
source cs612_venv/bin/activate
```

Then install some libraries.

```
pip install torch torchvision
```

You can then run the code.

```
python train_model.py
```

The code will download the training and testing data (if you run it for the first time) and train a model from scratch. The model is then saved as the file mnist.pt
