# Transforming Drug Development: Exploring the Power of Graph Representations in Predicting Cardiotoxicity

This project delves into the realm of graph representation learning to predict the cardiotoxicity of molecules. It investigates the efficacy of Graph Neural Networks (GNNs) against traditional non-graph Neural Networks (NNs) using the Tox21 toxicity dataset.

## Getting Started

To begin, clone the repository:

```
git clone https://github.com/anonymousoxford2024/Computational-Medicine-Exam.git
```

Or via SSH:

```
git clone git@github.com:anonymousoxford2024/Computational-Medicine-Exam.git
```

Next, create a virtual environment using Python version 3.9.6.

Install the required dependencies:

```
pip install -r requirements.txt
```

With the setup complete, you can proceed to either retrain or evaluate the models.

## Reproducing Results

To perform evaluation:

```
python src/evaluation/run_evaluation.py
```

To retrain the models:

```
python src/train_baseline.py
```

```
python src/train_gnn.py
``` 

This ensures you can replicate the results effectively.
