# SharedCon

### 1. Dataset preparation

for using shared semantics

```bash
python shared_semantics.py
```

If you want to specify the number of the shared semantics,
or adapt another sentence embedding model,
```bash
python shared_semantics.py \
--cluster_num 10 \
--load_dataset ihc_pure \
--load_sent_emb_model simcse
```

for proximal anchor experiment
```bash
python noisy_positive.py
```

### 2. Preprocessing
```bash
python preprocess_dataset.py
```


### 3. Training
Adjust the hyperparameters with ```train_config.py```
then execute ```train.py```

### 4. Testing
Adjust the directories and test datasets with ```eval_config.py```
then execute ```eval.py```