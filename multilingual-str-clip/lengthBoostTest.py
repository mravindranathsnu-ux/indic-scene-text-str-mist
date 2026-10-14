python3 -c "
with open('data/tamil/train/train.tsv', encoding='utf-8') as f:
    lens = [len(l.rstrip().split('\t',1)[1]) for l in f if '\t' in l]
print(f'avg={sum(lens)/len(lens):.1f} max={max(lens)} n_13plus={sum(1 for l in lens if l>=13)}')
"
