import json
from typing import Tuple
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from utils.utils import *
import yaml
import shutil
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def load_dataset(
    data_path: Path,
    labels_file: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    annotations_path = data_path / labels_file

    with open(annotations_path, "r") as f:
        targets_json = json.load(f)

    images_df = pd.DataFrame.from_records(targets_json["images"])
    images_df.rename(columns={"id": "image_id"}, inplace=True)
    images_df = images_df[["image_id", "file_name"]]

    targets_df = pd.DataFrame.from_records(targets_json["annotations"])
    targets_df[["xmin", "ymin", "w", "h"]] = targets_df["bbox"].tolist()
    targets_df["xmax"] = targets_df["xmin"] + targets_df["w"]
    targets_df["ymax"] = targets_df["ymin"] + targets_df["h"]
    targets_df.reset_index(inplace=True)
    targets_df['index'] = targets_df['id']
    targets_df.rename(
        columns={"index": "target_id", "category_id": "label_id"}, inplace=True
    )
    targets_df = targets_df[
        ["target_id", "image_id", "label_id", "xmin", "ymin", "xmax", "ymax"]
    ]

    return images_df, targets_df


np.random.seed(42)
def stratified_data_split(
        dataset: pd.DataFrame,
        x_columns: list,
        y_columns: list,
        train_fraction: float,
        n_splits: int
) -> Tuple[list, list, list]:

    if not 0 < train_fraction < 1:
        raise ValueError(f"Training fraction must be a value between 0 and 1, got {train_fraction}")

    x = dataset[x_columns].to_numpy()
    y = dataset[y_columns].to_numpy()
    kfold = IterativeStratification(
        n_splits=n_splits, order=2,
    )
    train_splits, val_splits, test_splits = [], [], []
    for train, test in kfold.split(X=x, y=y):
        test_indexes, everything_else_indexes = test, train
        x_test, x_else = x[test_indexes], x[everything_else_indexes]
        y_test, y_else = y[test_indexes, :], y[everything_else_indexes, :]
        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.1, 0.9])
        train_indexes, validation_indexes = next(stratifier.split(x_else, y_else))
        x_train, x_val = (x_else[train_indexes], x_else[validation_indexes])

        train_subset = pd.DataFrame(x_train, columns=x_columns)
        test_subset = pd.DataFrame(x_test, columns=x_columns)
        val_subset = pd.DataFrame(x_val, columns=x_columns)
        train_splits.append(train_subset)
        test_splits.append(test_subset)
        val_splits.append(val_subset)

    return train_splits, val_splits, test_splits


def convert_coco(labels_dir='../coco/annotations/'):

    save_dir = Path('yolo_labels')
    if save_dir.exists():
        shutil.rmtree(save_dir)
    for p in save_dir / 'labels', save_dir / 'images':
        p.mkdir(parents=True, exist_ok=True)


    for json_file in sorted(Path(labels_dir).resolve().glob('*.json')):
        fn = Path(save_dir) / 'labels' / json_file.stem.replace('instances_', '')  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        coco2yolo = {cat["id"]: i for i, cat in enumerate(data["categories"])}


        images = {f'{x["id"]:d}': x for x in data['images']}

        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)


        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
            img = images[f'{img_id:d}']
            h, w, f = img['height'], img['width'], img['file_name']

            bboxes = []
            for ann in anns:
                if ann['iscrowd']:
                    continue

                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = coco2yolo[ann['category_id']]
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            with open((fn / f).with_suffix('.txt'), 'a') as file:
                for i in range(len(bboxes)):
                    line = bboxes[i]
                    formatted_line = ' '.join(['%g' % num for num in line])
                    file.write(formatted_line + '\n')

    return


def create_yolo_dataset(data_path: Path, train_fraction: float = 0.6, n_splits = 5):

    yolo_labels = Path('./yolo_labels/')

    if yolo_labels.exists():
        pass
    else:

        convert_coco(labels_dir=data_path)


    shutil.move(yolo_labels / 'labels' / 'labels', data_path)
    (yolo_labels / 'labels').rmdir()
    (yolo_labels / 'images').rmdir()
    yolo_labels.rmdir()

    annotations_path = data_path / "labels.json"
    with open(annotations_path, "r") as f:
        targets_json = json.load(f)

    id2label = {cat["id"]: cat["name"] for cat in targets_json["categories"] if cat['supercategory'] != 'none'}
    print(id2label)
    data = pd.DataFrame.from_records(targets_json["images"])
    data.rename(columns={"id": "image_id"}, inplace=True)
    data = data[["image_id", "file_name"]]

    classes_df = pd.DataFrame.from_records(targets_json["annotations"])
    classes_df = classes_df[["image_id", "category_id"]]
    labels = []
    none_label_value = max(classes_df['category_id'].unique()) + 1

    data['label_None'] = 0
    for i in tqdm(data.index, desc='Preparing data'):
        image_id = data['image_id'].iloc[i]
        label_list = list(classes_df.query('image_id==@image_id')['category_id'])
        if len(label_list) == 0:
            label_list = [none_label_value]
            data.at[i, 'label_None'] = 1
        labels.append(label_list)
    data['labels'] = labels

    jobs_encoder = MultiLabelBinarizer()
    jobs_encoder.fit(data['labels'])
    transformed = jobs_encoder.transform(data['labels'])
    ohe_df = pd.DataFrame(transformed)
    ohe_df = ohe_df.add_prefix('label_')
    data = pd.concat([data, ohe_df], axis=1).drop(['labels'], axis=1)

    print('Starting data split\n')
    train_split, val_split, test_split = stratified_data_split(data, data.columns[:2], data.columns[2:], train_fraction, n_splits)

    print('Saving dataset into txt\n')
    split_index = 0
    for train, val, test in zip(train_split, val_split, test_split):
        split_index += 1
        yolo_train = f'yolo_train_{split_index}.txt'
        yolo_val = f'yolo_val_{split_index}.txt'
        yolo_test = f'yolo_test_{split_index}.txt'

        train['file_name'] = './images/' + train['file_name']
        test['file_name'] = './images/' + test['file_name']
        val['file_name'] = './images/' + val['file_name']
        np.savetxt('./' + str(data_path) + '/' + yolo_train, train['file_name'], fmt='%s')
        np.savetxt('./' + str(data_path) + '/' + yolo_val, val['file_name'], fmt='%s')
        np.savetxt('./' + str(data_path) + '/' + yolo_test, test['file_name'], fmt='%s')
        data_yaml = {'path': '../' + str(data_path),
                     'train': yolo_train,
                     'val': yolo_val,
                     'test': yolo_test,
                     'nc': len(id2label),
                     'names': list(id2label.values())
                     }
        with open(f'data_{split_index}.yaml', 'w') as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=None, sort_keys=False)
    print('Dataset is ready\n')

    return