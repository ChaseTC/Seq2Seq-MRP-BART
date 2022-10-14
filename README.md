# Sequence-to-Sequence Meaning Representation Parsing with BART
This is code-base for the paper `Sequence-to-Sequence Meaning Representation Parsing with BART`. It contains the code to preprocess the data, train the neural network models and evaluate the models.

Data is extracted using [this tool](https://gitlab.cs.uct.ac.za/jbuys/mrs-processing) using the deepbank semantic annotations for EDS. The data is located in `data/extracted`

## Environment Setup
Create a python virtual environment:
```
python3 -m venv myenv
```
Activate the environment and install the required modules:
```
source myenv/bin/activate
pip3 install -r requirements.txt
```
The virtual environment needs to be active before running any of the below scripts.

## Config
This project uses `Weights and Biases` to log training data and for hyperparameter search, to make use of this functionality it is necessary to add your Weights and Biases auth key and project name to `config\config.yaml`

## Preprocessing

### Convert to PENMAN
First run `convert_penman.py`
```
python3 preprocessing\convert_penman.py
```
This will convert the extracted EDS data into sentence-PENMAN pairs. Output will be located in `data/penman`.

### Remove Outliers
Following this run `remove_outliers.py`
```
python3 preprocessing\remove_outliers.py
```
This will remove sentence-PENMAN pairs that are longer than 512 tokens. Output will be located in `data/subset`.

## Neural-net
### Training
To train a model run the following script:
```
python3 neural-net/train.py
```
Hyperparameters and other settings can be changed in `config/config.yaml`

`train.py` takes the following optional parameters:
- `-c` or `--config` can be used to specify a path to a config file, defaults to `config/config.yaml`
- `-cp` or `--checkpoint` can be used to specify a path to a checkpoint to continue training from

The model will be saved in `models\model-name`

### Predicting
To use a model for prediction run the following:
```
python3 neural-net/predict.py [model]
```
where `[model]` is the path to your model.

This will write your model's predictions of the test data set to `data\predictions\[model]-pred.txt`

### Hyperparameter Search
Hyperparameter search was done using [Weights and Biases Sweeps](https://docs.wandb.ai/guides/sweeps)

`hp_search.py` was used for hyperparameter tuning, the search space and hyperparameters need to be manually changed in the code for use.

The search space (line 20 in `hp_search.py`) can be modified according to the following [documentation](https://docs.wandb.ai/guides/sweeps/configuration). Other hyperparameters can be changed in the base config (line 52-55) or in the Seq2SeqTrainingArgs (line 100)

Run using:
```
python3 neural-net\hp_search.py
```

`hp_search.py` takes the following optional parameter:
- `-c` or `--config` can be used to specify a path to a config file containing the WandB key, defaults to `config/config.yaml`

## Postprocessing and Evaluation
### Evaluation
To evaluate your model's predictions on the test data set run the following:
```
python3 postprocessing/eval.py [input]
```
Where `[input]` is the prediction file.

`eval.py` takes the following optional parameters:
- `--start_edm` this will use start EDM instead of EDM
- `--no_smatch` this will disable Smatch

Output will be written to `data\eval`

Our `bart-base-pt` models predictions are included in `data\predictions\base-pretrained.txt`

### parser.py
`parser.py` contains the LL(1) parser implementation used to correct bad predictions.

## Utils
`get_labels.py` is a script that was used to get the node and edge labels that were added to the tokenizer vocabulary