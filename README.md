# mlops-template

## How to install and run your application 

pip install uv

uv sync

uv run hello

## How to change your uv environment

Install python in uv with

uv python install <your version>

List pythons available with 

uv python list

Change your current version with

uv python pin <your version>

Install modules with

uv pip install yourmodule

Erase or create your venv with (with python version pinned, it will be used)

uv venv

Do not forget to activate your venv !!!

## Exemple mlflow workflow

https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/README.rst
https://mlflow.org/docs/latest/ml/projects/#building-workflows

## How to execute your MLproject in your local uv environment 

mlflow run . -P path=coucou --env-manager=local

or

mlflow run ./src/mlops_ses/training -P path=./data/output.csv --env-manager=local 

