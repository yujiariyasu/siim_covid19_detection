python tools/test.py configs/swin/swin003.py \
    ../mmdet-checkpoints/swin003_fold0.pth  ../data/covid/train_bbox_annotations_mmdet.pkl \
    --outer-fold 0 --out ../predictions/swin003_fold0.pkl --data-dir ../data/covid/train/ --work-dir '' --show-score-thr 0.01

python tools/test.py configs/swin/swin003.py \
    ../mmdet-checkpoints/swin003_fold1.pth  ../data/covid/train_bbox_annotations_mmdet.pkl \
    --outer-fold 1 --out ../predictions/swin003_fold1.pkl --data-dir ../data/covid/train/ --work-dir '' --show-score-thr 0.01

python tools/test.py configs/swin/swin003.py \
    ../mmdet-checkpoints/swin003_fold2.pth  ../data/covid/train_bbox_annotations_mmdet.pkl \
    --outer-fold 2 --out ../predictions/swin003_fold2.pkl --data-dir ../data/covid/train/ --work-dir '' --show-score-thr 0.01

python tools/test.py configs/swin/swin003.py \
    ../mmdet-checkpoints/swin003_fold3.pth  ../data/covid/train_bbox_annotations_mmdet.pkl \
    --outer-fold 3 --out ../predictions/swin003_fold3.pkl --data-dir ../data/covid/train/ --work-dir '' --show-score-thr 0.01

python tools/test.py configs/swin/swin003.py \
    ../mmdet-checkpoints/swin003_fold4.pth  ../data/covid/train_bbox_annotations_mmdet.pkl \
    --outer-fold 4 --out ../predictions/swin003_fold4.pkl --data-dir ../data/covid/train/ --work-dir '' --show-score-thr 0.01

python tools/test.py configs/swin/swin006.py \
    ../experiments/mmdetection/swin005/fold0/best_AP50_epoch_8.pth  ../data/covid/train_bbox_annotations_mmdet_multiclass.pkl \
    --outer-fold 0 --out ../predictions/swin006_fold0.pkl --data-dir ../data/covid/train/ --work-dir '' --show-score-thr 0.01

python tools/test.py configs/swin/swin006.py \
    ../experiments/mmdetection/swin005/fold1/best_AP50_epoch_7.pth  ../data/covid/train_bbox_annotations_mmdet_multiclass.pkl \
    --outer-fold 1 --out ../predictions/swin006_fold1.pkl --data-dir ../data/covid/train/ --work-dir '' --show-score-thr 0.01

python tools/test.py configs/swin/swin006.py \
    ../experiments/mmdetection/swin005/fold2/best_AP50_epoch_8.pth  ../data/covid/train_bbox_annotations_mmdet_multiclass.pkl \
    --outer-fold 2 --out ../predictions/swin006_fold2.pkl --data-dir ../data/covid/train/ --work-dir '' --show-score-thr 0.01

python tools/test.py configs/swin/swin006.py \
    ../experiments/mmdetection/swin005/fold3/best_AP50_epoch_8.pth  ../data/covid/train_bbox_annotations_mmdet_multiclass.pkl \
    --outer-fold 3 --out ../predictions/swin006_fold3.pkl --data-dir ../data/covid/train/ --work-dir '' --show-score-thr 0.01

python tools/test.py configs/swin/swin006.py \
    ../experiments/mmdetection/swin005/fold4/best_AP50_epoch_8.pth  ../data/covid/train_bbox_annotations_mmdet_multiclass.pkl \
    --outer-fold 4 --out ../predictions/swin006_fold4.pkl --data-dir ../data/covid/train/ --work-dir '' --show-score-thr 0.01
