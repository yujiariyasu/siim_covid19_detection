python main.py pseudoseg configs/seg/seg019.yaml --num-workers 4  \
    --checkpoint ../experiments/classify/seg019/sbn/fold0/checkpoints/epoch006-vm0.6304.ckpt \
    --data-dir ../data/covid/test_pngs/ --imgfiles public_test.txt \
    --act-fn sigmoid --save-preds-file ../predictions/public_test_0.csv \
    --save-pseudo-segs ../pseudosegs/public_test_0/


python main.py pseudoseg configs/seg/seg019.yaml --num-workers 4  \
    --checkpoint ../experiments/classify/seg019/sbn/fold1/checkpoints/epoch007-vm0.6295.ckpt \
    --data-dir ../data/covid/test_pngs/ --imgfiles public_test.txt \
    --act-fn sigmoid --save-preds-file ../predictions/public_test_1.csv \
    --save-pseudo-segs ../pseudosegs/public_test_1/


python main.py pseudoseg configs/seg/seg019.yaml --num-workers 4  \
    --checkpoint ../experiments/classify/seg019/sbn/fold2/checkpoints/epoch006-vm0.6348.ckpt \
    --data-dir ../data/covid/test_pngs/ --imgfiles public_test.txt \
    --act-fn sigmoid --save-preds-file ../predictions/public_test_2.csv \
    --save-pseudo-segs ../pseudosegs/public_test_2/


python main.py pseudoseg configs/seg/seg019.yaml --num-workers 4  \
    --checkpoint ../experiments/classify/seg019/sbn/fold3/checkpoints/epoch004-vm0.6174.ckpt \
    --data-dir ../data/covid/test_pngs/ --imgfiles public_test.txt \
    --act-fn sigmoid --save-preds-file ../predictions/public_test_3.csv \
    --save-pseudo-segs ../pseudosegs/public_test_3/


python main.py pseudoseg configs/seg/seg019.yaml --num-workers 4  \
    --checkpoint ../experiments/classify/seg019/sbn/fold4/checkpoints/epoch008-vm0.6255.ckpt \
    --data-dir ../data/covid/test_pngs/ --imgfiles public_test.txt \
    --act-fn sigmoid --save-preds-file ../predictions/public_test_4.csv \
    --save-pseudo-segs ../pseudosegs/public_test_4/
