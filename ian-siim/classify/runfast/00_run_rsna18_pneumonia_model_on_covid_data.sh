python main.py predict configs/mks/mk001.yaml \
    --num-workers 4 \
    --checkpoint ../experiments/classify/mk001/sbn/checkpoints/last.ckpt \
    --data-dir ../data/covid/train/ \
    --imgfiles covid_dicom_train_files.txt \
    --save-preds-file ../predictions/rsna18-model-on-covid-data/mk001.csv \
    --act-fn softmax

