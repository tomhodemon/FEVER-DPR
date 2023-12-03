K_values=(1 2 3 4 5 6 7 8 9 10 20 50 100 200)

for K in "${K_values[@]}"
do
    python eval_dpr.py \
        --k $K \
        --indexes indexes.bin \
        --split test \
        --query_encoder tomhodemon/fever-query_encoder-lora-bsz16-77588-gradacc1 \
        --passage_encoder tomhodemon/fever_passage_encoder-lora-bsz16-77588-gradacc1 \
        --output_dir results/bsz16-77588-gradacc1-lora \
        --lora
done