## ImJoy-powered Interactive Segmentation

This project enables deep learning powered interactive segmentation with ImJoy.

In contrast to traditional deep learning model training where all the annotations are collected before the training, interactive learning runs the model training while adding new annotations.
	
## Key feature
* Using ImJoy as an interface for data loading and annotation
* Track training progress and guide the model throughout training

Therefore, users can encourage the model to learn by feeding in appropriate data (eg. worse-performing samples).

## Installation
```bash
conda create -n iteractive-ml python=3.7.2 -y
conda activate iteractive-ml
pip install -r requirements.txt
python3 -m ipykernel install --user --name imjoy-iteractive-ml --display-name "ImJoy Iteractive ML"
```


## Usage

Start a jupyter notebook with ImJoy
```bash
imjoy --jupyter
```

You can download our example dataset to get started:
```bash
# this will save the example dataset to `./data/hpa_dataset_v2`
python download_example_dataset.py
```

Create a jupyter notebook and run the followin code in a cell:
```python
from imjoy_plugin import start_interactive_segmentation

start_interactive_segmentation("./data/hpa_dataset_v2", ["microtubules.png", "er.png", "nuclei.png"], object_name="cell", scale_factor=0.5, resume=True)
```

We also made a python notebook to illustrate the whole interactive training workflow in **tutorial.ipynb**
