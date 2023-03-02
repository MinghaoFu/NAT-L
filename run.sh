src=en # source language
tgt=de # target language

dataset=Europarl-sent.en-de
text=./data/${dataset} # data path
data_bin_dir=./data-bin/${dataset} # path to data-bin
save_dir=./checkpoint/Tr-${dataset} # save path
model_path=./maskPredict # path to MASKPREDICT
time=$(date +"%Y-%m-%d %T")

if [ $# -eq 0 ]; then
    echo "No arguments supplied."

elif [ $1 = "preprocess" ]; then
    echo 'Preprocessing data...'

    python preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref ${text}/train --validpref ${text}/valid \
    --testpref ${text}/test  --destdir ${data_bin_dir}  --workers 60  --joined-dictionary\
    #--srcdict ${model_path}/maskPredict_${src}_${tgt}/dict.${src}.txt \
    #--tgtdict ${model_path}/maskPredict_${src}_${tgt}/dict.${tgt}.txt \

elif [ $1 = "train" ]; then
    echo 'Training...'
    mkdir -p ${save_dir}
    srun -p NLP --gres=gpu:4 -N1 \
    python train.py ${data_bin_dir} --arch transformer_long --share-all-embeddings \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr 5e-4 \
    --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_long \
    --max-tokens 32768 --weight-decay 0.01 --fp16 --dropout 0.3 --encoder-layers 6 --encoder-embed-dim 512 \
    --decoder-layers 6 --decoder-embed-dim 512 --max-source-positions 10000 \
    --max-target-positions 10000 --max-update 300000 --seed 0 --save-dir ${save_dir} --keep-last-epochs 20 \
    > ${save_dir}/atrain.log 2>&1 &

elif [ $1 = "eval" ]; then
    if [ $2 = "average" ]; then
        echo "average checkpoint..."
        srun -p NLP --gres=gpu:1 --quotatype=spot \
        python scripts/average_checkpoints.py --inputs ${save_dir} --num-epoch-checkpoints 15 \
        --output ${save_dir}/checkpoint_average_best.pt 
    fi

    # srun -p NLP --gres=gpu:1 --quotatype=spot \
    # fairseq-generate ${data_bin_dir} --path ${save_dir}/checkpoint_average_best.pt --remove-bpe --max-sentences 20 --beam 5 \
    # --skip-invalid-size-inputs-valid-test \
    # > ${save_dir}/aeval.log 2>&1 &
    echo "evaluating..."
    srun -p NLP --gres=gpu:1 --quotatype=reserved \
    python generate.py ${data_bin_dir} --path ${save_dir}/checkpoint_average_best.pt \
    --task translation --remove-bpe --max-sentences 20 --lenpen 1 --skip-invalid-size-inputs-valid-test \
    > ${save_dir}/aeval.log 2>&1 &
else
    echo 'Unknown type.'
fi