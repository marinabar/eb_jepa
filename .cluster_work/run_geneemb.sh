#!/bin/bash
# Launch the Tahoe-100M gene-embedding build sharded across ALL 8 B200s.
# Sequences are pre-resolved into the shared seq_cache (rest_batch source), so each
# shard goes straight to GPU embedding (its own batch-resolve pass is then a no-op).
set -u
cd /data/eb_jepa
export HF_HOME=/data/hf
export HF_XET_HIGH_PERFORMANCE=1
PY=/data/evo2_venv/bin/python
OUT=/data/gene_emb_cache
NSHARDS=8
GPUS=(0 1 2 3 4 5 6 7)   # all 8 GPUs
MODEL=arcinstitute/evo2_7b_base

echo "=== gene-emb build start $(date) : $NSHARDS shards on GPUs ${GPUS[*]} ===" | tee -a /data/gene_emb.log
for s in $(seq 0 $((NSHARDS-1))); do
  g=${GPUS[$s]}
  log=/data/gene_emb_shard_${s}.log
  echo "launch shard $s on GPU $g -> $log" | tee -a /data/gene_emb.log
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" -m scripts.build_gene_embeddings \
      --data_dir /data/tahoe-100m --out_dir "$OUT" \
      --evo2_model "$MODEL" --num_shards "$NSHARDS" --shard_index "$s" \
      > "$log" 2>&1 &
  echo $! > /data/gene_emb_shard_${s}.pid
  sleep 6   # stagger model loads
done

echo "all shards launched; waiting..." | tee -a /data/gene_emb.log
wait
echo "=== all shards finished $(date); merging ===" | tee -a /data/gene_emb.log
CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.build_gene_embeddings \
    --out_dir "$OUT" --merge_only True --num_shards "$NSHARDS" \
    >> /data/gene_emb.log 2>&1
echo "=== merge done $(date) ===" | tee -a /data/gene_emb.log
if [ -f "$OUT/.DONE" ]; then
  echo "BUILD COMPLETE" | tee -a /data/gene_emb.log
  cat "$OUT/.DONE" | tee -a /data/gene_emb.log
else
  echo "MERGE DID NOT PRODUCE .DONE -- check logs" | tee -a /data/gene_emb.log
fi
