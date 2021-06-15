## Repository for "Modeling Hierarchical Structures with Continuous Recursive Neural Networks" (ICML 2021)

[Arxiv Link](https://arxiv.org/abs/2106.06038)

### Requirements

* torch                              1.7.1+cu110
* tqdm                               4.42.1
* numpy                              1.19.5
* hyperopt                           0.2.5
* jsonlines                          2.0.0
* torchtext                          0.8.1
* python                             3.7

### Data Setup
* Put the SNLI files (downloaded and extracted from [here](https://nlp.stanford.edu/projects/snli/)) in `data/NLI_data/SNLI/` 
* Put the MNLI files (downloaded and extracted from [here](https://cims.nyu.edu/~sbowman/multinli/)) in `data/NLI_data/MNLI/` 
* Put the Logical Inference data files (train0,train1,train2,.....test12) (downloaded from [here](https://github.com/yikangshen/Ordered-Memory/tree/master/data/propositionallogic)) in `data/NLI_data/PNLI/`
* Put the ListOps data files (downloaded from [here](https://github.com/yikangshen/Ordered-Memory/tree/master/data/listops)) in `data/Classifier_data/listops/` 
* Put the ListOps extrapolation data (downloaded from the url mentioned [here](https://github.com/facebookresearch/latent-treelstm/blob/master/data/listops/external/urls.txt)) in `data/Classifier_data/listops_extrapolation_data/`
* SST2 and SST5 data is downloaded and used through torchtext (so you don't have to do anything for SST5/2 data at this point)
* Put `glove.840B.300d.txt` (download `glove.840B.300d.zip` from [here](https://nlp.stanford.edu/projects/glove/)) in `embeddings/glove/`

### Preprocessing
Run `bash preprocess.sh` (after the data is set up)

### Notes
* FOCN is an older name for CRvNN. In Classifier/ the name "FOCN" is used, whereas in inference/ I updated the code and using "
CRvNN" as Model name. (also, "entropy" or "entropy_penalty" is an earlier name for "halt penalty")
  
* To see the main source code check `inference/models/encoders/CRvNN.py` (Eq. 5 version in paper) or `inference/models/encoders/CRvNN2.py` (Eq. 6 version in paper)

* The code framework in `inference/` is newer. If you want to build up from this repository, I would recommend mainly refering to `inference/`. The separate `classifier/` code is there for the sake of better reproducibility (because the reported classifier experiments were run using it).

### Hyperparamter Tuning
Should not be necessary to approximately reproduce the results because the chosen hyperparameters are already set up in the configs. But if you want to hyperparameter tune anyway then consider the following instructions.
* Hypertune CRvNN on Logical Inference: `python hypertrain.py --model=CRvNN --dataset=PNLI_LG` (after going inside `inference/`)
* Hypertune CRvNN on ListOps: `python hypertrain.py --model=FOCN --dataset=ListOps --limit=50000 --epochs=10` (after going inside `classifier/`)
* Hypertune CRvNN on SST5: `python hypertrain.py --model=FOCN --dataset=SST5 --limit=-1 --epochs=3` (after going inside `classifier/`)
* Hypertune CRvNN on SNLI: `python hypertrain.py --model=CRvNN --dataset=SNLI` (after going inside `inference/`)

Hyperparameter search space can be modified in `classifier/hyperconfigs/` and `inference/hyperconfigs/` (check the already existing "hyperconfigs" files there.)
Hyperparameters themselves can be modified in `classifier/configs/` and `inference/configs/` (check the already existing config files in there)

Note: You have to manually modify the configs according to the chosen hyperparameters after hyperparameter tuning.

### Training

* To run logical inference experiment (length generalization), go to `inference/` and run `python train.py --model=CRvNN --times=5 --dataset=PNLI_LG --model_type=NLI --limit=-1`
* To run logical inference experiment (split A systematicity), go to `inference/` and run `python train.py --model=CRvNN --times=5 --dataset=PNLI_A --model_type=NLI --limit=-1`
* Same as above for split B or split C systematicity test (just replace PNLI_A with PNLI_B or PNLI_C).
* To run ListOps experiment, go to `classifier/` and run `python train.py --model=FOCN --times=5 --dataset=ListOps --model_type=Classifier --limit=-1`
* You can run the same command as above but replace ListOps with SST5 or SST2 to run experiments on SST5 or SST2 respectively.
* To run SNLI experiments, go to `inference/` and run `python train.py --model=CRvNN --times=5 --dataset=SNLI --model_type=NLI --limit=-1`
* To run MNLI experiments, go to `inference/` and run `python train.py --model=CRvNN --times=3 --dataset=SNLI --model_type=NLI --limit=-1`
* To run the "speed test" in the paper, go to `classifier/` and run `python train.py --model=FOCN --times=1 --dataset=stress100 --model_type=Classifier --limit=-1` (stress100 runs it for sequence length of 81-100. For higher sequence lengths eg. 201-500 use stress500 and so on. See the argparse options in `classifier/parser.py`)
* To run the same test with ordered memory, use `--model=ordered_memory` above. 
* You can run the ablation tests by using the ablated model with `--model`. See the options in `parser.py` (within `inference/` or `classifier`). For now the ablations are set up only for ListOps and PNLI_LG (length generation test on logical inference)
* Model name (see the --model options in parser.py)  descriptions for ablation : In `inference/`, LR_CRvNN corresponds to CRvNN without structure. CRvNN_balanced uses a trivial balanced binary tree. CRvNN_LSTM uses LSTM cell. CRvNN_no_entropy removes the halt penalty mentioned in the paper. CRvNN_no_gelu uses ReLU instead of GeLU in the gated recursive cell. CRvNN_no_transtion removes the transition features mentioned in the paper. CRvNN_no_modulation removes modulated sigmoid. Same for FOCN named versions in `classifier/`.

### MNLI experiments note

Depending on how the dynamic halting behaves during the training you may get a complete run on MNLI without issues, or you may run into OOM errors. 

One nasty way to workaround is reduce the train_batch_size (in `inference/configs/MNLI_configs.py`). Note that gradient accumulation is on and the effective batch size after gradient accumulation is specific by batch_size in the config, so changing the train_batch_size for a epoch or two wouldn't make things too inconsistent (if you reduce the train_batch_size try to do that by a factor of 2).

There is also a bit dynamic train batch size adjustment going on according to the sequence length within `inference/collaters/NLI_collater.py` and you can play around with it a bit to avoid the OOM for the epoch (you can revert it back later).


A better solution could be to set an upperbound to the number of recursion. Currently the upperbound is the sequence size (S) - 2. A better upperbound may be `min(M, S-2)` where M can be the maximum upperbound that should not be crossed no matter the sequence size. There should be a good (i.e something that wouldn't cause accuracy loss) M value (for MNLI) given that the model can run fine on several complete runs suggesting that dynamic halting does not need to go beyond some M (going beyond which causes OOM). To change the upperbound, you have to change the source code `inference/models/encoders/CRvNN.py` (line 249) (replace S-2, with min(M,S-2) where M can be whatever you want to set; you can also pass the M parameter through configs (`inference/configs/`) and then simply set up conditlions for switching between using M and not using it or using different M values for different datasets). 


Note everything I said applies to my experience with training on a single gpu p3.2x AWS machine. Things may vary in other cases. In low resource settings, using a low M parameter to put a maximum limit to tree depth may be helpful (or not; I am not sure).


### Testing
Testing is usually automatically followed up after the training is complete. If you want to re-run test (given some inference weights are saved) without running training use the `--test=True`. 

### MNLI prediction
The trained MNLI wll be tested only on dev set. We can't directly test on the actual test set because the labels are hidden. To test on actual test set, we can generate its prediction on test set and upload it to kaggle (for MNLI mathced and MNLI mismatched).
To create a kaggle-compatible predictions file for MNLI test set that can be uploaded to Kaggle to get test scores, run `python predict_MNLI.py --model=CRvNN --initial_time=0` (In `inference/`). Initial_time specifies the run number (0 refers to first run, 1 may refer to second run etc. (That is, if you rain train.py with --times=5, the training will be repeated for 5 times with different random seeds and thus, different inference weights will be saved.  Here you can specify which inference weight run you are trying to predict on.))
Predictions will be generated in `inference/predictions/`

### Tree Extraction
* To extract trees you can run `python extract_trees.py --(whatever argparse commands you want to select a specific CRvNN model trained on a specific dataset)` in `inference/` (you can probably copy paste the code in `classifier/` and use it for classifier models as well)
* There is a variable list named `texts` in `inference/extract_trees.py` around line 87 which you can change to try to extract trees from different inputs.

### Credits
* `classifier/models/encoders/ordered_memory.py` or `inference/models/encoders/ordered_memory.py` is adapted from [here](https://github.com/yikangshen/Ordered-Memory).
* `classifier/optimizers/ranger.py` or `inference/optimizers/ranger.py` is adapted from [here](https://github.com/anoidgit/transformer/blob/master/optm/ranger.py).
* `classifier/optimizers/radam.py` or `inference/optimizers/radam.py` is adapted from [here](https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py).

### Contact
Feel free to contact me (write github issues, or reach out through emails, or both) if you run into any issues. 

### Cite

To be updated. 
