prepare: 
create jpg images https://www.kaggle.com/yujiariyasu/siim-covid19-convert-to-jpg-1536px?scriptVersionId=71593635
- set train folder to yuji_siim/input/images
The images are placed like this.. yuji_siim/input/images/train/xxx.jpg
- set train_with_size.csv folder to yuji_siim/input/
create train.csv https://www.kaggle.com/yujiariyasu/create-train-csv?scriptVersionId=71593668
- set train_for_classification.csv and train_for_detection.csv to yuji_siim/input
create segmentation mask https://www.kaggle.com/yujiariyasu/lungfield-segmentation-and-crop?scriptVersionId=71453959
- set mask_train folder to yuji_siim/input/
The segmentation masks are placed like this.. yuji_siim/input/mask_train/xxx.jpg

detection training:

```
cd yuji_siim
python yolo_train_one_fold.py -c mixup05_l6 --fold 0
python yolo_train_one_fold.py -c mixup05_l6 --fold 1
python yolo_train_one_fold.py -c mixup05_l6 --fold 2
python yolo_train_one_fold.py -c mixup05_l6 --fold 3
python yolo_train_one_fold.py -c mixup05_l6 --fold 4

python yolo_train_one_fold.py -c mixup05_l --fold 0
python yolo_train_one_fold.py -c mixup05_l --fold 1
python yolo_train_one_fold.py -c mixup05_l --fold 2
python yolo_train_one_fold.py -c mixup05_l --fold 3
python yolo_train_one_fold.py -c mixup05_l --fold 4
```

The model will be saved in the results folder like this.
`results/mixup05_l/fold_0/weights/best.pt`
The models are not included in this submission because they are too heavy, but they are in the kaggle dataset.
https://www.kaggle.com/yujiariyasu/yolo-models

wbf:
```
python wbf.py
```

classification training:
```
python train_one_fold.py -c model0changelr --fold 0
python train_one_fold.py -c model0changelr --fold 1
python train_one_fold.py -c model0changelr --fold 2
python train_one_fold.py -c model0changelr --fold 3
python train_one_fold.py -c model0changelr --fold 4

python train_one_fold.py -c swinmixupchangelr --fold 0
python train_one_fold.py -c swinmixupchangelr --fold 1
python train_one_fold.py -c swinmixupchangelr --fold 2
python train_one_fold.py -c swinmixupchangelr --fold 3
python train_one_fold.py -c swinmixupchangelr --fold 4
cd ..
```

The model will be saved in the results folder like this.
`results/model0changelr/fold0.pt`
The models are not included in this submission because they are too heavy, but they are in the kaggle dataset.
https://www.kaggle.com/yujiariyasu/model0changelr
https://www.kaggle.com/yujiariyasu/swinmixupchangelr

predict in this note
https://www.kaggle.com/yujiariyasu/fork-of-siim-covid-19-full-pipeline-v2?scriptVersionId=71599585
