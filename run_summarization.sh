python scripts/run_summarization.py \
    --model-checkpoint LazarusNLP/IndoNanoT5-base \
    --dataset-name LazarusNLP/indonlg \
    --dataset-config indosum \
    --input-column-name input \
    --target-column-name target \
    --input-max-length 512 \
    --target-max-length 512 \
    --num-beams 5 \
    --output-dir outputs/indo-nanot5-indosum \
    --num-train-epochs 5 \
    --optim adamw_torch_fused \
    --learning-rate 1e-3 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 16 \
    --hub-model-id LazarusNLP/IndoNanoT5-base-IndoSum

python scripts/run_summarization.py \
    --model-checkpoint LazarusNLP/IndoNanoT5-base \
    --dataset-name LazarusNLP/indonlg \
    --dataset-config liputan6_canonical \
    --input-column-name input \
    --target-column-name target \
    --input-max-length 512 \
    --target-max-length 512 \
    --num-beams 5 \
    --output-dir outputs/indo-nanot5-liputan6-canonical \
    --num-train-epochs 50 \
    --optim adamw_torch_fused \
    --learning-rate 1e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 16 \
    --hub-model-id LazarusNLP/IndoNanoT5-base-Liputan6-Canonical

# eval Canonical model on Extreme test set
python scripts/run_summarization.py \
    --model-checkpoint LazarusNLP/IndoNanoT5-base-Liputan6-Canonical \
    --dataset-name LazarusNLP/indonlg \
    --dataset-config liputan6_extreme \
    --input-column-name input \
    --target-column-name target \
    --input-max-length 512 \
    --target-max-length 512 \
    --num-beams 5 \
    --output-dir outputs/indo-nanot5-liputan6-extreme \
    --per-device-eval-batch-size 16 \
    --do-eval-only \
    --hub-model-id LazarusNLP/IndoNanoT5-base-Liputan6-Extreme