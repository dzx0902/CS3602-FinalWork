python src/benchmark.py --mode flash --max-new-tokens 128
python src/compute_ppl.py --mode flash --dataset wikitext
python src/compute_ppl.py --mode flash --dataset pg19
