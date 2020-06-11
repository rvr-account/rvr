# Representations via Representations

Representation via Representations is a project aimed at improving transfer learning to out-of-distribution examples. Motivated by the challenge of finding robust biomedical predictors of disease, the model leverages data from heterogenous sources to discover feature representations that allow for accurate prediction outside of the training data.

This codebase owes its foundations to David Madras, Elliot Creager, Toni Pitassi, Richard Zemel in their paper Learning Adversarially Fair and Transferable Representations (https://arxiv.org/abs/1802.06309). Github link: https://github.com/VectorInstitute/laftr

## setting up a project-specific virtual env
```
mkdir ~/venv 
python3 -m venv ~/venv/rvr
```
where `python3` points to python 3.6.X. Then
```
source ~/venv/rvr/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
or 
```
pip install -r requirements-gpu.txt
```
for GPU support

## running a single fair classification experiment
```
source simple_example.sh
```
The bash script first trains LAFTR and then evaluates by training a naive classifier on the LAFTR representations (encoder outputs).

## running a sweep of fair classification with various hyperparameter values
```
python src/generate_sweep.py sweeps/mnist_sweep/sweep.json
source sweeps/mnist_sweep/command.sh
```
For a bigger sweep call `src/generate_sweep` with `sweeps/mnist_sweep/sweep.json`, or design your own sweep config. Then all the generated scripts can be run in parallel.
The parameters for the model are found in `sweeps/mnist_sweep/config.json` and model dimensions can be found in `conf/templates/model/mnist.json`

## data
The synthetic datasets are provided in `data/runhet/*.npz` or `data/runorfunc/*.npz`
Due to space constraint, all the dataset generation files for synthetic data and colored MNIST can be found in `src/data_processing/`. The dataset generation code for PACS is pulled from https://github.com/ameroyer/TFDatasets.

## model

The implementations of the models can be found in `src/codebase/models.py`
The network architecture itself is found in `src/codebase/mlp.py`. See the paper for exact details
