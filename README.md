# Yuji's part
​
### Hardware
- CPU / Intel Xeon Platinum 8360Y x 2
- GPU / NVIDIA A100 SXM4 x 1
- Memory / 512GiB
- NVMe SSD / Intel SSD DC P4510 2.0TB x 2
- Interconnect / InfiniBand HDR x 4
​
### OS
CentOS v7.5
​
### How to train your model
prepare: 
create jpg images https://www.kaggle.com/yujiariyasu/siim-covid19-convert-to-jpg-1536px?scriptVersionId=71593635
- set train folder to yuji_siim/input/
The images are placed like this.. yuji_siim/input/train/xxx.jpg
- set train_with_size.csv folder to yuji_siim/input/
create train.csv https://www.kaggle.com/yujiariyasu/create-train-csv?scriptVersionId=71593668
- set train_for_classification.csv and train_for_detection.csv to yuji_siim/input
create segmentation mask https://www.kaggle.com/yujiariyasu/lungfield-segmentation-and-crop?scriptVersionId=71453959
- set mask_train folder to yuji_siim/input/
The segmentation masks are placed like this.. yuji_siim/input/mask_train/xxx.jpg
​
detection:
​
```
cd yuji_siim
python yolo_train_one_fold.py -c mixup05_l6 --fold 0
python yolo_train_one_fold.py -c mixup05_l6 --fold 1
python yolo_train_one_fold.py -c mixup05_l6 --fold 2
python yolo_train_one_fold.py -c mixup05_l6 --fold 3
python yolo_train_one_fold.py -c mixup05_l6 --fold 4
​
python yolo_train_one_fold.py -c mixup05_l --fold 0
python yolo_train_one_fold.py -c mixup05_l --fold 1
python yolo_train_one_fold.py -c mixup05_l --fold 2
python yolo_train_one_fold.py -c mixup05_l --fold 3
python yolo_train_one_fold.py -c mixup05_l --fold 4
```
​
wbf:
```
python wbf.py
```
​
classification:
```
python train_one_fold.py -c model0changelr --fold 0
python train_one_fold.py -c model0changelr --fold 1
python train_one_fold.py -c model0changelr --fold 2
python train_one_fold.py -c model0changelr --fold 3
python train_one_fold.py -c model0changelr --fold 4
​
python train_one_fold.py -c swinmixupchangelr --fold 0
python train_one_fold.py -c swinmixupchangelr --fold 1
python train_one_fold.py -c swinmixupchangelr --fold 2
python train_one_fold.py -c swinmixupchangelr --fold 3
python train_one_fold.py -c swinmixupchangelr --fold 4
cd ..
```
​
### How to make predictions on a new test set.
Register the two models we created for detection and two models for classification into the kaggle dataset, and register Ian's models. And run the kernel for inference.
https://www.kaggle.com/yujiariyasu/fork-of-siim-covid-19-full-pipeline-v2?scriptVersionId=71599585
​
# Ian's part
​
### Hardware
​- CPU / Intel Xeon Gold 6242 @ 2.80GHz
- GPU / NVIDIA Quadro RTX 6000 24 GB x4
- RAM / 64 GB 

### OS
​RedHat v7.7

### Setup
`cd ian-siim/ ; bash setup.sh ; cd mmdetection ; pip install -r requirements.txt ; pip install . -v -e`
​
Download the RSNA Pneumonia Detection Challenge dataset from Kaggle. Place it in `data/rsna18/`. There should now be a folder named `data/rsna18/stage_2_train_images/`.

Download the SIIM-FISABIO-RSNA COVID-19 Detection dataset from Kaggle. place it in `data/covid/`.

```
cd ian-siim/etl
python 01_convert_covid_dicoms_to_pngs.py
```

### How to train your model

## Pretraining
```
cd ian-siim/classify
python main.py train configs/mks/mk030.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm
python main.py train configs/mks/mk032.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm


cd ian-siim/detect 
python main.py train configs/rsna/rsna002.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm


cd ian-siim/mmdetection 
bash tools/dist_train.sh configs/swin/swin_rsna002.py 4
```

## Classification
```
cd ian-siim/classify

python main.py train configs/seg/seg019.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

python main.py train configs/seg/seg019.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

python main.py train configs/seg/seg019.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

python main.py train configs/seg/seg019.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

python main.py train configs/seg/seg019.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

python main.py train configs/seg/seg032.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

python main.py train configs/seg/seg032.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

python main.py train configs/seg/seg032.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

python main.py train configs/seg/seg032.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

python main.py train configs/seg/seg032.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4
```

## Detection (EfficientDet)

```
cd ian-siim/detect

python main.py train configs/mks/mk004.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

python main.py train configs/mks/mk004.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

python main.py train configs/mks/mk004.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

python main.py train configs/mks/mk004.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

python main.py train configs/mks/mk004.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

python main.py train configs/mks/mk007.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

python main.py train configs/mks/mk007.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

python main.py train configs/mks/mk007.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

python main.py train configs/mks/mk007.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

python main.py train configs/mks/mk007.yaml --num-workers 4 \
    --gpus 4 --num_nodes 1 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4
```

## Detection (MMDetection)

```
cd ian-siim/mmdetection

bash tools/dist_train.sh configs/swin/swin004.py 4 --kfold 0
bash tools/dist_train.sh configs/swin/swin004.py 4 --kfold 1
bash tools/dist_train.sh configs/swin/swin004.py 4 --kfold 2
bash tools/dist_train.sh configs/swin/swin004.py 4 --kfold 3
bash tools/dist_train.sh configs/swin/swin004.py 4 --kfold 4
```

### How to make predictions on a new test set.
​See above.