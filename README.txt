conda env create -f environment.yml
conda activate bd4h

########### to start replicating, assuming datasets are present in bd4h/data/*
# python text2diag.py 

# download raw data from https://physionet.org/content/mimiciii-demo/1.4/
# save data into bd4h/data to match dir set up

# conda deactivate
# conda env remove -n bd4h