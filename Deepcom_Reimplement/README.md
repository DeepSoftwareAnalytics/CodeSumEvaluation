## Deep code comment generation
* paper: https://xin-xia.github.io/publication/icpc182.pdf
* official code: null

##  Environment

```
conda create -n deepcom_env python=3.6 
conda activate deepcom_env
conda install pytorch-gpu=1.3.1
pip install git+https://github.com/casics/spiral.git
pip install nltk==3.2.5  ipdb==0.13.3 javalang==0.12.0 networkx==2.3 prettytable

```

## Data processing

### Step1 get data
```
cd data/tlcodesum
bash get_data.sh
```
### Step2 code and summary processing
```
cd data/tlcodesum
bash preprocess.sh
```


## Run

```
cd code/
bash run_tlcodesum.sh $gpus $seed
```

For example:
```
bash run_tlcodesum.sh 0 0
```
## Evaluate

```
refs_filename=../saved_model/test.gold
preds_filename=../saved_model/test.pred
python evaluate.py --refs_filename $refs_filename --preds_filename $preds_filename 

```


## Result


| Best epoch |   BLEU-DM  |GPU|
| :--------- | :----:  |:----: |
|98|40.1778|v100|





