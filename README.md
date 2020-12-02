## COSMO: Conditional SEQ2SEQ-based Mixture Model for Zero-Shot Commonsense Question Answering

This repo contains the source code for the paper [COSMO: Conditional SEQ2SEQ-based Mixture Model for Zero-ShotCommonsense Question Answering](https://arxiv.org/abs/2011.00777).

### Requirements
* Python
* PyTorch
* Fairseq

### Experiments

To start the experiments, first run the following script to download the atomic data:
'''
bash setup/get_atomic.sh
'''

Then run the following scripts to prepare the atomic data:
'''
python setup/prep_data.py
'''

