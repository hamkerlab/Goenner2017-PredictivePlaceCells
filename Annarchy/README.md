# Hippocampus model - ANNarchy

The neuron definitions are in `ann_neuron_model.py`. Simulation parameters are in `ann_define_params.py`. 

## Simulation scripts

* `ann_learning_test_sequence.py` : Simulates a sequence based on a learned Context-to-DG weight pattern (saved in `data/learned_weights_DGinp`).

* `ann_pftask_recoded_seqtest.py` : Main function for starting full spatial learning simulation. (Note: At the start of the first trial, there might not be a sequence happening).
