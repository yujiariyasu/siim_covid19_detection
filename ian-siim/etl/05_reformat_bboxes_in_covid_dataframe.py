import ast
import pandas as pd


def change_box_format(boxes):
    if str(boxes) == 'nan': return boxes
    boxes = ast.literal_eval(boxes)
    boxes = [(b['x'], b['y'], b['width'], b['height']) for b in boxes]
    return boxes


df = pd.read_csv('../data/covid/train_kfold_cleaned.csv')
df['boxes'] = df.boxes.apply(lambda x: change_box_format(x))
df.to_csv('../data/covid/train_kfold_cleaned_w_bboxes.csv', index=False)