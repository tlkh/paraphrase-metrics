# if running this script directly from container
# assume folder mounted at /workspace
#cd /workspace/paraphrase-metrics
#pip3 install -r requirements.txt
#cd /workspace/paraphrase-metrics/classifier

python3 train.py -m microsoft/deberta-base -d MRPC_COR -seed 1
python3 train.py -m microsoft/deberta-base -d MRPC_COR -seed 2
python3 train.py -m microsoft/deberta-base -d MRPC_COR -seed 3
python3 train.py -m microsoft/deberta-base -d MRPC_COR -seed 4
python3 train.py -m microsoft/deberta-base -d MRPC_COR -seed 5

rm -rf ./results/mrpc_cor/deberta-base/output_*/checkpoint-*

python3 train.py -m microsoft/deberta-base -d MRPC -seed 1
python3 train.py -m microsoft/deberta-base -d MRPC -seed 2
python3 train.py -m microsoft/deberta-base -d MRPC -seed 3
python3 train.py -m microsoft/deberta-base -d MRPC -seed 4
python3 train.py -m microsoft/deberta-base -d MRPC -seed 5

rm -rf ./results/mrpc/deberta-base/output_*/checkpoint-*

python3 train.py -m microsoft/deberta-large -d MRPC_COR -seed 1
python3 train.py -m microsoft/deberta-large -d MRPC_COR -seed 2
python3 train.py -m microsoft/deberta-large -d MRPC_COR -seed 3
python3 train.py -m microsoft/deberta-large -d MRPC_COR -seed 4
python3 train.py -m microsoft/deberta-large -d MRPC_COR -seed 5

rm -rf ./results/mrpc_cor/deberta-large/output_*/checkpoint-*

python3 train.py -m microsoft/deberta-large -d MRPC -seed 1
python3 train.py -m microsoft/deberta-large -d MRPC -seed 2
python3 train.py -m microsoft/deberta-large -d MRPC -seed 3
python3 train.py -m microsoft/deberta-large -d MRPC -seed 4
python3 train.py -m microsoft/deberta-large -d MRPC -seed 5

rm -rf ./results/mrpc/deberta-large/output_*/checkpoint-*

