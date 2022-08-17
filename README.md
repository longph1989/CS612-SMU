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
pip install torch torchvision autograd scipy matplotlib
```

You can then follow the instructions in each exercise.
