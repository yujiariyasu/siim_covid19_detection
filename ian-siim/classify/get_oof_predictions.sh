python main.py test configs/seg/seg032.yaml --num-workers 4 --kfold 0 \
    --checkpoint ../experiments/classify/seg032/sbn/fold0/checkpoints/epoch005-vm0.6223.ckpt \
    --data-dir ../data/covid/train_pngs/ --act-fn sigmoid --save-preds-file ../predictions/seg032/fold0.csv


python main.py test configs/seg/seg032.yaml --num-workers 4 --kfold 1 \
    --checkpoint ../experiments/classify/seg032/sbn/fold1/checkpoints/epoch009-vm0.6422.ckpt \
    --data-dir ../data/covid/train_pngs/ --act-fn sigmoid --save-preds-file ../predictions/seg032/fold1.csv


python main.py test configs/seg/seg032.yaml --num-workers 4 --kfold 2 \
    --checkpoint ../experiments/classify/seg032/sbn/fold2/checkpoints/epoch005-vm0.6394.ckpt \
    --data-dir ../data/covid/train_pngs/ --act-fn sigmoid --save-preds-file ../predictions/seg032/fold2.csv


python main.py test configs/seg/seg032.yaml --num-workers 4 --kfold 3 \
    --checkpoint ../experiments/classify/seg032/sbn/fold3/checkpoints/epoch005-vm0.6206.ckpt \
    --data-dir ../data/covid/train_pngs/ --act-fn sigmoid --save-preds-file ../predictions/seg032/fold3.csv


python main.py test configs/seg/seg032.yaml --num-workers 4 --kfold 4 \
    --checkpoint ../experiments/classify/seg032/sbn/fold4/checkpoints/epoch006-vm0.6052.ckpt \
    --data-dir ../data/covid/train_pngs/ --act-fn sigmoid --save-preds-file ../predictions/seg032/fold4.csv

