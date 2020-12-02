## COSMO: Conditional SEQ2SEQ-based Mixture Model for Zero-Shot Commonsense Question Answering

This repo contains the source code for the paper [COSMO: Conditional SEQ2SEQ-based Mixture Model for Zero-ShotCommonsense Question Answering](https://arxiv.org/abs/2011.00777).

### Requirements
* Python
* PyTorch
* Fairseq

### Experiments

To start the experiments, first run the following script to download the atomic data:

```
bash setup/get_atomic.sh
```

Then run the following scripts to prepare the atomic data:

```
python setup/prep_data.py
```


To generate binary data files for s2s model run the following script:

```
fairseq-preprocess \
--user-dir ./src \
--task prophetnet_cosmo \
--source-lang src --target-lang tgt \
--trainpref data/atomic/trn --validpref data/atomic/dev --testpref data/atomic/tst \
--destdir data/atomic/processed --srcdict data/vocab.txt --tgtdict data/vocab.txt \
--workers 20
```

To fine-tune the model on atomic run the following script:

```
DATA_DIR=data/atomic/processed
USER_DIR=./src
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=data_atomic/save_dir/
TENSORBOARD_LOGDIR=data/atomic/tensorboard
PRETRAINED_MODEL=data/pretrained_checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt

fairseq-train \
--fp16 \
--user-dir $USER_DIR --task prophetnet_cosmo --arch $ARCH \
--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
--lr 0.0001 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--criterion $CRITERION --label-smoothing 0.1 \
--update-freq 32  --max-sentences 2 \
--num-workers 4 \
--load-from-pretrained-model $PRETRAINED_MODEL \
--load-sep \
--ddp-backend=no_c10d --max-epoch 10 \
--max-source-positions 512 --max-target-positions 512 \
--skip-invalid-size-inputs-valid-test \
--seed 1 \
--save-dir $SAVE_DIR \
--keep-last-epochs 10 \
--tensorboard-logdir $TENSORBOARD_LOGDIR \
$DATA_DIR
```
