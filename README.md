[![launch ImJoy](https://imjoy.io/static/badge/launch-imjoy-badge.svg)](https://imjoy.io/#/app?workspace=kaibu&plugin=https://raw.githubusercontent.com/imjoy-team/imjoy-interactive-segmentation/master/interactive_trainer.py)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/imjoy-team/imjoy-interactive-segmentation/master?filepath=Tutorial.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imjoy-team/imjoy-interactive-segmentation/blob/master/Tutorial.ipynb)

## ImJoy-powered Interactive Segmentation

This project enables deep learning powered interactive segmentation with ImJoy.

In contrast to traditional deep learning model training where all the annotations are collected before the training, interactive learning runs the model training while adding new annotations.
	
## Key feature
* Using ImJoy as an interface for data loading and annotation
* Track training progress and guide the model throughout training

Therefore, users can encourage the model to learn by feeding in appropriate data (eg. worse-performing samples).

## Installation
```bash
conda create -n interactive-ml python=3.7.2 -y
conda activate interactive-ml

git clone https://github.com/CellProfiling/imjoy-interactive-segmentation.git
cd imjoy-interactive-segmentation
pip install -r requirements.txt
python3 -m ipykernel install --user --name imjoy-interactive-ml --display-name "ImJoy Interactive ML"
```


## Usage

Start a the jupyter notebook server with ImJoy
```bash
imjoy --jupyter
```

Importantly, create a notebook file with kernel spec named "ImJoy Interactive ML".


You can download our example dataset to get started:
```bash
# this will save the example dataset to `./data/hpa_dataset_v2`
python download_example_dataset.py
```

Create a jupyter notebook and run the followin code in a cell:
```python
from imjoy_plugin import start_interactive_segmentation
from models.interactive_cellpose import CellPoseInteractiveModel

model = CellPoseInteractiveModel('./data/hpa_dataset_v2/__models__',
                                 channels=[2, 3],
                                 style_on=0,
                                 default_diameter=100)

start_interactive_segmentation(model,
                               "./data/hpa_dataset_v2",
                               ["microtubules.png", "er.png", "nuclei.png"],
                               mask_type="labels",
                               object_name="cell",
                               scale_factor=1.0,
                               resume=True)
```

We also made a python notebook to illustrate the whole interactive training workflow in **tutorial.ipynb**
