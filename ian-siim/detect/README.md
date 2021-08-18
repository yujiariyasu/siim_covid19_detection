Create base datasets:

```
python 00_create_cv_splits.py
python 01_create_segmentation_dataset.py
```

Train model to segment CVC, NGT, ETT:

```
python main.py train configs/seg/seg005.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 0
python main.py train configs/seg/seg005.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 1
python main.py train configs/seg/seg005.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 2
python main.py train configs/seg/seg005.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 3
python main.py train configs/seg/seg005.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 4
```

Get OOF predictions for segmented images. 

```
python main.py segment_test configs/seg/seg005.yaml --checkpoint '../experiments/seg005/sbn/fold0/checkpoints/epoch=018-vm=0.7560.ckpt' --data-dir ../data/train/ --act-fn sigmoid --num-workers 4 --save-preds-file ../data/predicted-segmentations-png2/ --kfold 0
python main.py segment_test configs/seg/seg005.yaml --checkpoint '../experiments/seg005/sbn/fold1/checkpoints/epoch=019-vm=0.7529.ckpt' --data-dir ../data/train/ --act-fn sigmoid --num-workers 4 --save-preds-file ../data/predicted-segmentations-png2/ --kfold 1
python main.py segment_test configs/seg/seg005.yaml --checkpoint '../experiments/seg005/sbn/fold2/checkpoints/epoch=018-vm=0.7506.ckpt' --data-dir ../data/train/ --act-fn sigmoid --num-workers 4 --save-preds-file ../data/predicted-segmentations-png2/ --kfold 2
python main.py segment_test configs/seg/seg005.yaml --checkpoint '../experiments/seg005/sbn/fold3/checkpoints/epoch=018-vm=0.7547.ckpt' --data-dir ../data/train/ --act-fn sigmoid --num-workers 4 --save-preds-file ../data/predicted-segmentations-png2/ --kfold 3
python main.py segment_test configs/seg/seg005.yaml --checkpoint '../experiments/seg005/sbn/fold4/checkpoints/epoch=019-vm=0.7528.ckpt' --data-dir ../data/train/ --act-fn sigmoid --num-workers 4 --save-preds-file ../data/predicted-segmentations-png2/ --kfold 4
```


Get segmentation predictions for unsegmented images:
```
python -m torch.distributed.launch --nproc_per_node=4 main.py segment configs/seg/seg005.yaml --checkpoint '../experiments/seg005/sbn/fold0/checkpoints/epoch=018-vm=0.7560.ckpt,../experiments/seg005/sbn/fold1/checkpoints/epoch=019-vm=0.7529.ckpt,../experiments/seg005/sbn/fold2/checkpoints/epoch=018-vm=0.7506.ckpt,../experiments/seg005/sbn/fold3/checkpoints/epoch=018-vm=0.7547.ckpt,../experiments/seg005/sbn/fold4/checkpoints/epoch=019-vm=0.7528.ckpt' --data-dir ../data/train/ --imgfiles ../data/unsegmented_images.txt --act-fn sigmoid --num-workers 4 --save-preds-file ../data/predicted-segmentations-png2/ --accelerator ddp
```

python -m torch.distributed.launch --nproc_per_node=4 main.py segment configs/seg/seg005.yaml --checkpoint '../experiments/seg005/sbn/fold0/checkpoints/epoch=018-vm=0.7560.ckpt,../experiments/seg005/sbn/fold1/checkpoints/epoch=019-vm=0.7529.ckpt,../experiments/seg005/sbn/fold2/checkpoints/epoch=018-vm=0.7506.ckpt,../experiments/seg005/sbn/fold3/checkpoints/epoch=018-vm=0.7547.ckpt,../experiments/seg005/sbn/fold4/checkpoints/epoch=019-vm=0.7528.ckpt' --data-dir ../data/train/ --imgfiles test.txt --data-dir ../data/test/ --act-fn sigmoid --num-workers 4 --save-preds-file ../data/predicted-segmentations-png-test/ --accelerator ddp

Train 11-class classification models:
```
python main.py train configs/mks/mk040.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 0
python main.py train configs/mks/mk040.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 1
python main.py train configs/mks/mk040.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 2
python main.py train configs/mks/mk040.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 3
python main.py train configs/mks/mk040.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 4

python main.py train configs/mks/mk039.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 0 --seed 87
python main.py train configs/mks/mk039.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 1 --seed 87
python main.py train configs/mks/mk039.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 2 --seed 87
python main.py train configs/mks/mk039.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 3 --seed 87
python main.py train configs/mks/mk039.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 4 --seed 87

python main.py train configs/mks/mk022.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 0 --seed 86
python main.py train configs/mks/mk022.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 1 --seed 86
python main.py train configs/mks/mk022.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 2 --seed 86
python main.py train configs/mks/mk022.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 3 --seed 86
python main.py train configs/mks/mk022.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm --kfold 4 --seed 86
```

```
python -m torch.distributed.launch --nproc_per_node=4 main.py test configs/mks/mk016.yaml \
    --checkpoint '../experiments/mk016/sbn/fold0/checkpoints/epoch=017-vm=0.9518.ckpt' \
    --act-fn sigmoid \
    --save-preds-file ../predictions/mk016/fold0.csv \
    --accelerator ddp \
    --data-dir ../data/train \
    --kfold 0 

python -m torch.distributed.launch --nproc_per_node=4 main.py test configs/mks/mk019.yaml \
    --checkpoint '../experiments/mk019/sbn/fold0/checkpoints/epoch=008-vm=0.9543.ckpt' \
    --act-fn sigmoid \
    --save-preds-file ../predictions/mk019/fold0.csv \
    --accelerator ddp \
    --data-dir ../data \
    --kfold 0 

python -m torch.distributed.launch --nproc_per_node=4 main.py test configs/mks/mk016.yaml \
    --checkpoint '../experiments/mk016/sbn/fold1/checkpoints/epoch=019-vm=0.9556.ckpt' \
    --act-fn sigmoid \
    --save-preds-file ../predictions/mk016/fold1.csv \
    --accelerator ddp \
    --data-dir ../data/train \
    --kfold 1 

python -m torch.distributed.launch --nproc_per_node=4 main.py test configs/mks/mk019.yaml \
    --checkpoint '../experiments/mk019/sbn/fold1/checkpoints/epoch=009-vm=0.9601.ckpt' \
    --act-fn sigmoid \
    --save-preds-file ../predictions/mk019/fold1.csv \
    --accelerator ddp \
    --data-dir ../data \
    --kfold 1

python -m torch.distributed.launch --nproc_per_node=4 main.py test configs/mks/mk016.yaml \
    --checkpoint '../experiments/mk016/sbn/fold2/checkpoints/epoch=018-vm=0.9589.ckpt' \
    --act-fn sigmoid \
    --save-preds-file ../predictions/mk016/fold2.csv \
    --accelerator ddp \
    --data-dir ../data/train \
    --kfold 2 

python -m torch.distributed.launch --nproc_per_node=4 main.py test configs/mks/mk019.yaml \
    --checkpoint '../experiments/mk019/sbn/fold2/checkpoints/epoch=006-vm=0.9606.ckpt' \
    --act-fn sigmoid \
    --save-preds-file ../predictions/mk019/fold2.csv \
    --accelerator ddp \
    --data-dir ../data \
    --kfold 2

python -m torch.distributed.launch --nproc_per_node=4 main.py test configs/mks/mk016.yaml \
    --checkpoint '../experiments/mk016/sbn/fold3/checkpoints/epoch=019-vm=0.9575.ckpt' \
    --act-fn sigmoid \
    --save-preds-file ../predictions/mk016/fold3.csv \
    --accelerator ddp \
    --data-dir ../data/train \
    --kfold 3

python -m torch.distributed.launch --nproc_per_node=4 main.py test configs/mks/mk019.yaml \
    --checkpoint '../experiments/mk019/sbn/fold3/checkpoints/epoch=006-vm=0.9619.ckpt' \
    --act-fn sigmoid \
    --save-preds-file ../predictions/mk019/fold3.csv \
    --accelerator ddp \
    --data-dir ../data \
    --kfold 3

python -m torch.distributed.launch --nproc_per_node=4 main.py test configs/mks/mk016.yaml \
    --checkpoint '../experiments/mk016/sbn/fold4/checkpoints/epoch=019-vm=0.9539.ckpt' \
    --act-fn sigmoid \
    --save-preds-file ../predictions/mk016/fold4.csv \
    --accelerator ddp \
    --data-dir ../data/train \
    --kfold 4

python -m torch.distributed.launch --nproc_per_node=4 main.py test configs/mks/mk019.yaml \
    --checkpoint '../experiments/mk019/sbn/fold4/checkpoints/epoch=007-vm=0.9605.ckpt' \
    --act-fn sigmoid \
    --save-preds-file ../predictions/mk019/fold4.csv \
    --accelerator ddp \
    --data-dir ../data \
    --kfold 4
```

```
python main.py full_inference configs/inf/inf000_fold0.yaml \
    --imgfiles fold0.txt \
    --data-dir ../data/train/ \
    --save-preds-file ../predictions/inf000/fold0.csv \
    --num-workers 4

python main.py full_inference configs/inf/inf000_fold1.yaml \
    --imgfiles fold1.txt \
    --data-dir ../data/train/ \
    --save-preds-file ../predictions/inf000/fold1.csv \
    --num-workers 4

python main.py full_inference configs/inf/inf000_fold2.yaml \
    --imgfiles fold2.txt \
    --data-dir ../data/train/ \
    --save-preds-file ../predictions/inf000/fold2.csv \
    --num-workers 4

python main.py full_inference configs/inf/inf000_fold3.yaml \
    --imgfiles fold3.txt \
    --data-dir ../data/train/ \
    --save-preds-file ../predictions/inf000/fold3.csv \
    --num-workers 4

python main.py full_inference configs/inf/inf000_fold4.yaml \
    --imgfiles fold4.txt \
    --data-dir ../data/train/ \
    --save-preds-file ../predictions/inf000/fold4.csv \
    --num-workers 4


python main.py full_inference configs/inf/inf000_5fold.yaml \
    --imgfiles fold0.txt \
    --data-dir ../data/train/ \
    --save-preds-file ../predictions/inf000/fold0_5fold.csv \
    --num-workers 4

python main.py full_inference configs/inf/inf000_5fold.yaml \
    --imgfiles test.txt \
    --data-dir ../data/test/ \
    --save-preds-file ../data/test_pseudolabels.csv \
    --num-workers 4

python main.py full_inference configs/inf/inf005_cxr14.yaml \
    --imgfiles cxr14_1.txt \
    --data-dir ../../cxr14/images \
    --save-preds-file ../predictions/inf005/cxr14_1.csv \
    --num-workers 4


python main.py full_inference configs/inf/inf005_cxr14.yaml \
    --imgfiles cxr14_2.txt \
    --data-dir ../../cxr14/images \
    --save-preds-file ../predictions/inf005/cxr14_2.csv \
    --num-workers 4


python main.py full_inference configs/inf/inf005_cxr14.yaml \
    --imgfiles cxr14_3.txt \
    --data-dir ../../cxr14/images \
    --save-preds-file ../predictions/inf005/cxr14_3.csv \
    --num-workers 4


python main.py full_inference configs/inf/inf005_cxr14.yaml \
    --imgfiles cxr14_all.txt \
    --data-dir ../../cxr14/images \
    --save-preds-file ../predictions/inf005/cxr14_all.csv \
    --num-workers 4
```

```
python main.py full_inference configs/inf/inf001_5fold.yaml \
    --imgfiles test.txt \
    --data-dir ../data/test/ \
    --save-preds-file ../data/test_inf001.csv \
    --num-workers 4
```

Train model to detect presence of CVC, NGT, ETT: 

`python main.py train configs/lid/lid001.yaml --benchmark --precision 16 --gpus 4 --num-workers 4 --accelerator ddp --sync_batchnorm`


Use line finder model to pseudo-label CheXpert:

```
python -m torch.distributed.launch --nproc_per_node=4 main.py predict configs/lid/lid001.yaml \
    --checkpoint '../experiments/lid001/sbn/checkpoints/epoch=004-vm=0.9914.ckpt' \
    --act-fn sigmoid \
    --data-dir ../data/ --imgfiles ../data/chexpert_AP_images.txt \
    --save-preds-file ../data/chexpert_line_finder_preds.pkl \
    --accelerator ddp
```

Get CheXpert images without CVC:

```
python -m torch.distributed.launch --nproc_per_node=4 main.py segment configs/seg/seg001.yaml \
    --checkpoint '../experiments/seg001/fold0/checkpoints/epoch=009-vm=0.7382.ckpt' \
    --act-fn sigmoid \
    --data-dir ../data/ --imgfiles ../data/chexpert_no_cvc_unsegmented_ranzcr.txt \
    --save-preds-file ../data/pseudosegmentations/ \
    --accelerator ddp
```


Use segmentation model to pseudo-label unsegmented images from RANZCR-CLiP and CheXpert images without CVC:


Train hybrid classification-segmentation model on segmented RANZCR-CLiP images and pseudo-labeled images from above, 5-fold:


Generate predicted out-of-fold segmentations for each RANZCR-CLiP image: 





