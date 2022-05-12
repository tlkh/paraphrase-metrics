# if running this script directly from container
# assume folder mounted at /workspace
#cd /workspace/paraphrase-metrics
#pip3 install -r requirements.txt
#cd /workspace/paraphrase-metrics/classifier

python3 train.py -m microsoft/deberta-base -d PAWS -bs 32 -warmup 0.1 -log 500 -gradaccum 4 -seed 1
python3 train.py -m microsoft/deberta-base -d PAWS -bs 32 -warmup 0.1 -log 500 -gradaccum 4 -seed 2
python3 train.py -m microsoft/deberta-base -d PAWS -bs 32 -warmup 0.1 -log 500 -gradaccum 4 -seed 3
python3 train.py -m microsoft/deberta-base -d PAWS -bs 32 -warmup 0.1 -log 500 -gradaccum 4 -seed 4
python3 train.py -m microsoft/deberta-base -d PAWS -bs 32 -warmup 0.1 -log 500 -gradaccum 4 -seed 5

rm -rf ./results/paws/deberta-base/output_*/checkpoint-*

python3 train.py -m microsoft/deberta-large -d PAWS -bs 32 -warmup 0.1 -log 500 -gradaccum 4 -seed 1
python3 train.py -m microsoft/deberta-large -d PAWS -bs 32 -warmup 0.1 -log 500 -gradaccum 4 -seed 2
python3 train.py -m microsoft/deberta-large -d PAWS -bs 32 -warmup 0.1 -log 500 -gradaccum 4 -seed 3
python3 train.py -m microsoft/deberta-large -d PAWS -bs 32 -warmup 0.1 -log 500 -gradaccum 4 -seed 4
python3 train.py -m microsoft/deberta-large -d PAWS -bs 32 -warmup 0.1 -log 500 -gradaccum 4 -seed 5

rm -rf ./results/pasw/deberta-large/output_*/checkpoint-*

