## General info
This project is an interactive deep learning framework that allows users to track and adjust the performance by adding new training data. In contrast to traditional deep learning model training where all data are loaded in the beginning and performance is measured at the end of training time/after convergence, we trained a model continuously in the cloud(?) or on local computer while monitoring the model progress and annotating new data in web interface (powered by [https://imjoy.io]). 
	
## Key feature
* Using ImJoy as an interface for customized data upload and labelling
* Track training progress and guide the model throughout training
* Throughout training, the model sees the most recent data and groundtruth. Therefore, users can encourage the model to learn by feeding in appropriate data (eg. worse-performing samples).

## Example of use
We made a python notebook to illustrate the whole interactive training workflow in **tutorial.ipynb**