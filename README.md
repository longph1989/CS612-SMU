# CS612-SMU

## Installation

First please make sure you have installed Python. Then install virtual enviroment and activate it.

- For Windows:

```
pip install virtualenv
python3 -m virtualenv cs612_venv
.\cs612_venv\Scripts\activate
```

- For MacOS:

```
pip install virtualenv
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
