#dlen=150
clen=50
slen=400
#dvoc=36202
cvoc=30000
svoc=30000
# code processing
# djl dfp dsi dlc dr
# summary processing
# cfp  csi  cfd

# code
is_djl=1
is_dfp=0
is_dsi=1
is_dlc=1
is_dr=1
# summary
is_cfp=0
is_csi=0
is_clc=1
# dup
in_trn_dup_r=0.0
acr_trn_tst_dup_r=0.0
in_val_dup_r=None
in_tst_dup_r=None
acr_trn_val_dup_r=0.0
# sample way
sbt_type=2
in_dup_type=1
acr_dup_type=0
#
code_params=djl${is_djl}_dfp${is_dfp}_dsi${is_dsi}_dlc${is_dlc}_dr${is_dr}
summary_params=cfp${is_cfp}_csi${is_csi}_cfd0_clc${is_clc}
sbt_params=sbt${sbt_type}
ast_params=ast

#data_params=${code_params}_${summary_params}_${sbt_params}
#data_dir=../data/tlcodesum/processed/$data_params
data_dir=../data/tlcodesum/processed/cfp0_csi0_cfd0_clc1_sbt2


batch_size=256
#code_dim=256
summary_dim=256
sbt_dim=256
epoch=200
rnn_hidden_size=256
learning_rate=0.001
seed=$2


params=clen${clen}-slen${slen}-cvoc${cvoc}-svoc${svoc}-bs${batch_size}-cdim${summary_dim}-sdim${sbt_dim}-rhs${rnn_hidden_size}-lr${learning_rate}-e${epoch}

current_time=$(date "+%Y%m%d%H%M%S")
output_dir=../saved_model/tlcodesum/random/$params/${current_time}
# output_dir=../saved_model/tlcodesum/$params/tmp
# output_dir=../saved_model/tlcodesum/$params/tmp # for debugging
mkdir -p $output_dir

# load_model_dir=$output_dir/checkpoint-last # for continuing training
load_model_dir=$output_dir/model-best-bleu  # for testing
#--debug  --step_log_freq 1
#--step_log_freq 1000

function train () {

echo "============TRAINING============"
CUDA_VISIBLE_DEVICES=$1 python -W ignore run.py --step_log_freq 1000 -epoch $epoch \
   --do_train --do_eval --do_test \
   -clc  \
    --seed $seed \
   -clen $clen -slen $slen \
   -cvoc $cvoc -svoc $svoc \
   --output_dir $output_dir \
   --data_dir $data_dir \
   -batch_size $batch_size \
   -summary_dim $summary_dim -sbt_dim $sbt_dim \
   -rnn_hidden_size $rnn_hidden_size \
   -lr $learning_rate \
    2>&1| tee $output_dir/train.log
}


function test () {

echo "============TESTING============"

CUDA_VISIBLE_DEVICES=$1 python -W ignore run.py --do_test \
   -clc \
   --load_model_dir $load_model_dir \
   -clen $clen -slen $slen \
   -cvoc $cvoc -svoc $svoc \
   --output_dir $output_dir \
   --data_dir $data_dir \
   -batch_size $batch_size \
   -summary_dim $summary_dim -sbt_dim $sbt_dim \
   -rnn_hidden_size $rnn_hidden_size \
   -lr $learning_rate \
   -epoch $epoch 2>&1| tee $output_dir/test.log

}

train $1
#test $1


refs_filename=$output_dir/test.gold
preds_filename=$output_dir/test.pred
python evaluate.py --refs_filename $refs_filename --preds_filename $preds_filename 2>&1| tee $output_dir/score.log


