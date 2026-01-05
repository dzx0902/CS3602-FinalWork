mode     | model       | B | prompt_seq_len | max_new_tokens | TTFT       | TPOT      | Throughput   | PeakMem
---------+-------------+---+----------------+----------------+------------+-----------+--------------+--------------
baseline | pythia-2.8b | 1 | 2454           | 128            | 314.645 ms | 32.796 ms | 30.491 tok/s | 20,746.41 MiB
baseline | pythia-2.8b | 1 | 2454           | 512            | 33.717 ms  | 33.208 ms | 30.114 tok/s | 21,140.94 MiB
baseline | pythia-2.8b | 1 | 2454           | 1024           | 37.654 ms  | 34.815 ms | 28.724 tok/s | 21,785.79 MiB
baseline | pythia-2.8b | 1 | 2454           | 2048           | 33.199 ms  | 37.968 ms | 26.338 tok/s | 23,070.73 MiB
flash    | pythia-2.8b | 1 | 2454           | 128            | 147.180 ms | 27.424 ms | 36.465 tok/s | 12,031.40 MiB
flash    | pythia-2.8b | 1 | 2454           | 512            | 66.252 ms  | 20.674 ms | 48.369 tok/s | 12,350.70 MiB
flash    | pythia-2.8b | 1 | 2454           | 1024           | 57.399 ms  | 19.134 ms | 52.263 tok/s | 12,835.35 MiB
flash    | pythia-2.8b | 1 | 2454           | 2048           | 55.696 ms  | 20.652 ms | 48.422 tok/s | 13,806.26 MiB