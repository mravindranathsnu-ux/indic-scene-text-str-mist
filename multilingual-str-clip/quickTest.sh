python lexicon_rerank_eval.py \
    --ckpt checkpoints/tamil/best.pt --clip_ckpt weights/ViT-B-16.pt \
    --test_ann data/tamil/test/test.tsv --test_image_root data/tamil/test/images \
    --lexicon_ann data/tamil/train/train.tsv data/tamil/val/val.tsv data/tamil/indiccorp_lexicon.tsv \
    --beam_size 5 --length_penalty 1.0 --no_tie_break_by_frequency \
    --out_csv results/tamil_lexicon_rerank.csv --yes
