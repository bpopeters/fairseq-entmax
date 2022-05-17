#!/bin/bash

# Train default fairseq transformer model for IWSLT14 (de-en)

readonly SEED=42
readonly PATIENCE=10

train(){
    local -r CP="$1"; shift
    fairseq-train data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --patience "${PATIENCE}"\
    --save-dir "${CP}" \
    --seed "${SEED}"
}

test_tokenized(){
    local -r CP="$1"; shift
    fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path "${CP}"/checkpoint_best.pt \
    --beam 5 \
    --source-lang de \
    --target-lang en \
    --remove-bpe \
    --scoring bleu
}

# computing sacrebleu using fairseq
test_detokenized(){
    local -r CP="$1"; shift
    fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path "${CP}"/checkpoint_best.pt \
    --beam 5 \
    --source-lang de \
    --target-lang en \
    --tokenizer moses \
    --remove-bpe \
    --scoring sacrebleu
}

test_sacrebleu(){
    local -r CP="$1"; shift
    fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path "${CP}"/checkpoint_best.pt \
    --beam 5 \
    --source-lang de \
    --target-lang en \
    --tokenizer moses \
    --remove-bpe \
    --scoring sacrebleu | tee /tmp/gen.out

    grep ^D /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
    grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
    sacrebleu  /tmp/gen.out.ref -i /tmp/gen.out.sys -m bleu -b -w 4
}



main(){
    #train "/home/antoniofarinhas/fairseq-beam/checkpoints/${SEED}-iwslt14-deen"
    #test_tokenized "/home/antoniofarinhas/fairseq-beam/checkpoints/${SEED}-iwslt14-deen"
    #test_detokenized "/home/antoniofarinhas/fairseq-beam/checkpoints/${SEED}-iwslt14-deen"
    test_sacrebleu "/home/antoniofarinhas/fairseq-beam/checkpoints/${SEED}-iwslt14-deen"
}


main