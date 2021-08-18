python main.py train configs/mks/mk008.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

python main.py train configs/mks/mk003.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

python main.py train configs/mks/mk003.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

python main.py train configs/mks/mk003.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

python main.py train configs/mks/mk003.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

python main.py train configs/rsna/rsna002.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

python main.py train configs/mks/mk001.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

python main.py train configs/mks/mk001.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

python main.py train configs/mks/mk001.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

python main.py train configs/mks/mk001.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4
