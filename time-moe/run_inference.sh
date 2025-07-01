python run_eval_ref.py -d /home/sa53869/time-series/datasets/time-moe-eval/synthetic_sinusoidal.csv -p 64 -c 512

python run_eval.py -d /home/sa53869/time-series/datasets/time-moe-eval/synthetic_sinusoidal.csv --column_index 0 -p 64 -c 2048 --max_windows 100 

python run_eval_with_si.py -d /home/sa53869/time-series/datasets/time-moe-eval/synthetic_sinusoidal.csv -p 64 -c 2048 --max_samples 100 

python run_eval_with_si_v2.py -d /home/sa53869/time-series/datasets/time-moe-eval/ETT-small/ETTm2.csv -p 64 -c 2048 --max_batches 100


python run_eval_prune.py -d dataset/ETT-small/ETTm2.csv -p 96 -c 512

python run_eval_prune.py -d dataset/ETT-small/ETTm2.csv -p 96 -c 1024

python run_eval_prune.py -d dataset/ETT-small/ETTm2.csv -p 96 -c 2048 -b 16

python run_eval_prune.py -d dataset/ETT-small/ETTm2.csv -p 96 -c 4096 -b 8