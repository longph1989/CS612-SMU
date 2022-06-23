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
pip install torch torchvision autograd sicpy matplotlib
```

You can then run the code.

```
python train_model.py
```

The code will download the training and testing data (if you run it for the first time) and train a model from scratch. The model is then saved as the file mnist.pt

You can then attack the previously trained model by running the code.

```
python attack_model.py
```

It will transform the PyTorch model into an internal representation and then use scipy to minimize an objective function. When the objective function is below 0, an adversarial sample is found.
