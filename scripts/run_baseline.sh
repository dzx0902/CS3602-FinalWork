python src/benchmark.py --mode baseline --max-new-tokens 128
python src/compute_ppl.py --mode baseline --dataset wikitext
python src/compute_ppl.py --mode baseline --dataset pg19
