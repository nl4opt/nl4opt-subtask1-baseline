# Subtask-1 (NER) Baseline Tutorial

Subtask-1 (NER) Baseline Tutorial

This pretrained baseline transformer model is adapted from a [popular NER competition](https://multiconer.github.io/) that leverages XLM-RoBERTa-base model.


## Setting up the code environment

### Clone the repo from git

Run the following command to clone the repository containing the scripts to create the baseline model. The relevant scripts will be in the directory ./baseline whereas data for the competition is under the directory ./data.
The current repository is empty and all relevant baseline scripts will be released by July 1st on AoE timezone.
`git clone https://github.com/nl4opt/nl4opt-competition.git`

### Setting up environment using conda

Create a conda environment for the NER task and install the required packages:
`conda create --name nl4opt-ner python=3.9.12`
`conda activate nl4opt-ner`
`cd nl4opt-competition/ner_task/baseline`
`pip install -r requirements.txt`

## Overview for Running the code for the NL4Opt Baseline

This code repository provides you with baseline approach for Named Entity Recognition (NER). In this repository the following functionalities are provided:

- CoNLL data readers
- Usage of any HuggingFace pre-trained transformer models
- Training and Testing through Pytorch-Lightning  

Please see below for a more detailed description on how to use this code is provided.

### Arguments:
The arguments are as follows:
- `--train` points to the path of the training data. In the case of the baseline model, this would be ../data/train/train.txt
- `--dev` points to the path of the training data. In the case of the baseline model, this would be ../data/dev/dev.txt
- `--out_dir` points to the path to save the output files.
- `--iob_tagging` can be adjusted if the tagging scheme is altered. However, this must be converted back to the 'conll' tagging scheme prior to evaluation.
- `--max_instances` is the maximum number of instances - we recommend to keep this untouched.
- `--max_length` is the maximum number of tokens per instance - we recommend to keep this untouched.
- `--encoder_model` is the pretrained encoder to select. The baseline model was trained on xlm-roberta-base but there are many others to choose from.
- `--model` is the path to the model. This is used for fine-tuning and evaluation. 
- `--model_name` is name to save your model under. 
- `--stage` is the stage of training. Either 'fit', 'validate', 'test', or 'predict'.
- `--prefix` is the prefix for storing evaluation files.
- `--batch_size` is the number of samples in a batch. 
- `--epochs` is the number of epochs to train for.
- `--lr` is the learning rate.
- `--dropout` is the dropout rate used.
- `--gpus` is the number of gpus to use.
- `--cuda` is the cuda device.
- `--epochs` is the number of epochs to train for.
- `--accum_grad_batches` is the number of batches before accumlating gradients.

```
p.add_argument('--train', type=str, help='Path to the train data.', default=None)
p.add_argument('--dev', type=str, help='Path to the dev data.', default=None)

p.add_argument('--out_dir', type=str, help='Output directory.', default='.')
p.add_argument('--iob_tagging', type=str, help='IOB tagging scheme', default='conll')

p.add_argument('--max_instances', type=int, help='Maximum number of instances', default=1500)
p.add_argument('--max_length', type=int, help='Maximum number of tokens per instance.', default=100)

p.add_argument('--encoder_model', type=str, help='Pretrained encoder model to use', default='xlm-roberta-large')
p.add_argument('--model', type=str, help='Model path.', default=None)
p.add_argument('--model_name', type=str, help='Model name.', default=None)
p.add_argument('--stage', type=str, help='Training stage', default='fit')
p.add_argument('--prefix', type=str, help='Prefix for storing evaluation files.', default='test')

p.add_argument('--batch_size', type=int, help='Batch size.', default=128)
p.add_argument('--accum_grad_batches', type=int, help='Number of batches for accumulating gradients.', default=1)
p.add_argument('--gpus', type=int, help='Number of GPUs.', default=1)
p.add_argument('--cuda', type=str, help='Cuda Device', default='cuda:0')
p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=5)
p.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
p.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)
``` 

## Train and evaluate the model

Execute the following commands to train, fine-tune, and evaluate the model on the development set. Participants will not have access to the test set that is used for model evaluation and leaderboard updates.

There are many approaches to training a neural network. However, the baseline framework provided does this in a two-step approach: first with a training session where the model is trained with early stopping; second with a fine-tuning session (potentially with a lower LR, or adaptive LR etc). 

### Training the model

`python train_model.py --train ../data/train/train.txt --dev ../data/dev/dev.txt --out_dir ./trained_model --model_name xlmr_lr_0.0001 --gpus 1 --epochs 25 --encoder_model xlm-roberta-base --batch_size 64 --lr 0.0001`

### Fine-tuning the model

`python fine_tune.py --train ../data/train/train.txt --out_dir ./trained_model --model_name xlmr_lr_0.0001 --gpus 1 --epochs 15 --encoder_model xlm-roberta-base --batch_size 64 --lr 0.0001 --model ./trained_model/xlmr_lr_0.0001/lightning_logs/version_0`

### Evaluating the model on the dev set

`python evaluate.py --test ../data/dev/dev.txt --out_dir ./trained_model --model_name xlmr_lr_0.0001 --gpus 1 --encoder_model xlm-roberta-base --batch_size 64 --model ./trained_model/xlmr_lr_0.0001/lightning_logs/version_1`


## Results

The model was evaluated on the reserved three new domains (production, science, and transportation) which are not represented in the released train or dev dataset (see rows 1-3 below). It was then also evaluated on 50 reserved samples from the same domain as those released to the participants ("Original Domains" below). Finally, the model was evaluated on the entire test set made up of all reserved sampled described above ("Entire Test Set" below) on which it achieved a micro-averaged F1 score of 0.8126. The leaderboard and final standings will only consider the micro-averaged F1 score (right-most column) of the submitted models on the entire test set. This model achieved the following F1 scores:
|                                | CONST<br/>DIR | LIMIT  | OBJ<br/>DIR | OBJ<br/>NAME | PARAM  | VAR    | MICRO<br/>AVG |
| ------------------------------ | ------------- | ------ | ----------- | ------------ | ------ | ------ | ------------- |
| **Production<br>(n=49)**       | 0.8205        | 0.9262 | 1.000       | 0.2295       | 0.9567 | 0.7122 | 0.8068        |
| **Science<br>(n=50)**          | 0.8759        | 0.8992 | 0.6667      | 0.000        | 0.8249 | 0.7881 | 0.7890        |
| **Transportation<br>(n=50)**   | 0.8083        | 0.8757 | 0.7619      | 0.0899       | 0.9034 | 0.7881 | 0.7849        |
| **Original Domains<br>(n=50)** | 0.8764        | 0.8808 | 0.8571      | 0.8286       | 0.9215 | 0.8270 | 0.8659        |
| **Entire Test Set<br>(n=199)** | 0.8411        | 0.8933 | 0.8679      | 0.4947       | 0.9012 | 0.7792 | 0.8126*       |
*\* Value that will be reported on the leaderboards page and used for the final evaluation when determining the winners.*

## License 
The code under this repository is licensed under the Apache 2.0 License found in this repository.