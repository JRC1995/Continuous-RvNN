cd classifier
python preprocess/process_ListOps.py
python preprocess/process_SST2.py
python preprocess/process_SST5.py
python preprocess/process_stress.py
cd ..
cd inference
python preprocess/process_PNLI_LG.py
python preprocess/process_PNLI_A.py
python preprocess/process_PNLI_B.py
python preprocess/process_PNLI_C.py
python preprocess/process_SNLI.py
python preprocess/process_MNLI.py