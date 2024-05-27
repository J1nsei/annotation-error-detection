from typing import Dict, Set, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torchvision

TARGETS_DF_COLUMNS = [
    "target_id",
    "image_id",
    "label_id",
    "xmin",
    "ymin",
    "w",
    "h",
]
PREDS_DF_COLUMNS = [
    "pred_id",
    "image_id",
    "label_id",
    "xmin",
    "ymin",
    "w",
    "h",
    "score",
]
ERRORS_DF_COLUMNS = ["pred_id", "target_id", "error_type"]

BACKGROUND_IOU_THRESHOLD = 0.1
FOREGROUND_IOU_THRESHOLD = 0.5


class ErrorType:
    OK = "correct"
    CLS = "classification"
    LOC = "localization"
    CLS_LOC = "cls & loc"
    DUP = "duplicate"
    BKG = "background"
    MISS = "missed"

def classify_predictions_errors(
        targets_df: pd.DataFrame,
        preds_df: pd.DataFrame,
        images_df: pd.DataFrame,
        iou_background: float = BACKGROUND_IOU_THRESHOLD,
        iou_foreground: float = FOREGROUND_IOU_THRESHOLD,
) -> pd.DataFrame:


    assert (set(TARGETS_DF_COLUMNS) - set(targets_df.columns)) == set()
    assert (set(PREDS_DF_COLUMNS) - set(preds_df.columns)) == set()

    pred2error = dict()
    target2pred = (
        dict()
    )
    pred2target = dict()
    missed_targets = set()


    preds_df = preds_df.sort_values(by="score", ascending=False)

    for image_id, im_preds_df in tqdm(preds_df.groupby("image_id"), desc='Classify errors'):

        im_targets_df = targets_df.query("image_id == @image_id").reset_index(
            drop=True
        )
        im_preds_df = im_preds_df.reset_index(drop=True)

        if im_targets_df.empty:
            pred2error = {**pred2error, **_process_empty_image(im_preds_df)}
        else:
            iou_matrix, iou_label_match_matrix = _compute_iou_matrices(
                im_targets_df, im_preds_df
            )


            for pred_idx in range(len(im_preds_df)):
                match_found = _match_pred_to_target_with_same_label(
                    pred_idx,
                    pred2error,
                    pred2target,
                    target2pred,
                    iou_label_match_matrix,
                    im_targets_df,
                    im_preds_df,
                    iou_background,
                    iou_foreground,
                )
                if match_found:
                    continue

                _match_pred_wrong_label_or_background(
                    pred_idx,
                    pred2error,
                    pred2target,
                    iou_matrix,
                    im_targets_df,
                    im_preds_df,
                    iou_background,
                    iou_foreground,
                )

    missed_targets = _find_missed_targets(targets_df, pred2target)
    errors_df = _format_errors_as_dataframe(
        pred2error, pred2target, missed_targets, targets_df, preds_df, images_df
    )
    return errors_df


def _process_empty_image(im_preds_df: pd.DataFrame) -> Dict[int, str]:

    return {
        pred_id: ErrorType.BKG for pred_id in im_preds_df["pred_id"].unique()
    }


def _compute_iou_matrices(
        im_targets_df: pd.DataFrame, im_preds_df: pd.DataFrame
) -> Tuple[np.array, np.array]:
    target_boxes = im_targets_df[["xmin", "ymin", "w", "h"]].values
    target_boxes[:, 2] += target_boxes[:, 0]
    target_boxes[:, 3] += target_boxes[:, 1]

    pred_boxes = im_preds_df[["xmin", "ymin", "w", "h"]].values
    pred_boxes[:, 2] += pred_boxes[:, 0]
    pred_boxes[:, 3] += pred_boxes[:, 1]

    iou_matrix = torchvision.ops.box_iou(
        torch.from_numpy(target_boxes),
        torch.from_numpy(pred_boxes),
    ).numpy()


    label_match_matrix = (
            im_targets_df["label_id"].values[:, None]
            == im_preds_df["label_id"].values[None, :]
    )

    iou_label_match_matrix = iou_matrix * label_match_matrix
    return iou_matrix, iou_label_match_matrix


def _match_pred_to_target_with_same_label(
        pred_idx: int,
        pred2error: Dict[int, str],
        pred2target: Dict[int, int],
        target2pred: Dict[int, int],
        iou_label_match_matrix: np.array,
        im_targets_df: pd.DataFrame,
        im_preds_df: pd.DataFrame,
        iou_background: float,
        iou_foreground: float,
) -> bool:

    target_idx = np.argmax(iou_label_match_matrix[:, pred_idx])
    iou = np.max(iou_label_match_matrix[:, pred_idx])
    target_id = im_targets_df.at[target_idx, "target_id"]
    pred_id = im_preds_df.at[pred_idx, "pred_id"]

    matched = False
    if iou >= iou_foreground:
        pred2target[pred_id] = target_id

        if target2pred.get(target_id) is None:
            target2pred[target_id] = pred_id
            pred2error[pred_id] = ErrorType.OK
        else:
            pred2error[pred_id] = ErrorType.DUP
        matched = True

    elif iou_background <= iou < iou_foreground:
        pred2target[pred_id] = target_id
        pred2error[pred_id] = ErrorType.LOC
        matched = True
    return matched


def _match_pred_wrong_label_or_background(
        pred_idx: int,
        pred2error: Dict[int, str],
        pred2target: Dict[int, int],
        iou_matrix: np.array,
        im_targets_df: pd.DataFrame,
        im_preds_df: pd.DataFrame,
        iou_background: float,
        iou_foreground: float,
) -> None:

    target_idx = np.argmax(iou_matrix[:, pred_idx])
    iou = np.max(iou_matrix[:, pred_idx])
    target_id = im_targets_df.at[target_idx, "target_id"]
    pred_id = im_preds_df.at[pred_idx, "pred_id"]

    if iou < iou_background:
        pred2error[pred_id] = ErrorType.BKG


    elif iou >= iou_foreground:
        pred2target[pred_id] = target_id
        pred2error[pred_id] = ErrorType.CLS
    else:

        pred2error[pred_id] = ErrorType.CLS_LOC


def _find_missed_targets(
        im_targets_df: pd.DataFrame, pred2target: Dict[int, int]
) -> Set[int]:

    matched_targets = [t for t in pred2target.values() if t is not None]
    missed_targets = set(im_targets_df["target_id"]) - set(matched_targets)
    return missed_targets


def _format_errors_as_dataframe(
        pred2error: Dict[int, str],
        pred2target: Dict[int, int],
        missed_targets: Set[int],
        targets_df,
        preds_df,
        images_df
) -> pd.DataFrame:

    errors_df = pd.DataFrame.from_records(
        [
            {"pred_id": pred_id, "error_type": error}
            for pred_id, error in pred2error.items()
        ]
    )
    errors_df["target_id"] = np.nan
    errors_df.set_index("pred_id", inplace=True)
    for pred_id, target_id in pred2target.items():
        errors_df.at[pred_id, "target_id"] = target_id

    missed_df = pd.DataFrame(
        {
            "pred_id": np.nan,
            "error_type": ErrorType.MISS,
            "target_id": list(missed_targets),
        }
    )
    errors_df = pd.concat(
        [errors_df.reset_index(), missed_df], ignore_index=True
    ).astype(
        {"pred_id": pd.Int64Dtype(), "target_id": pd.Int64Dtype(), "error_type": pd.StringDtype()}
    )
    errors_df['file_name'] = None
    for index, row in errors_df.iterrows():
        if pd.notna(row['pred_id']):
            id = row['pred_id']
            image_id = preds_df.loc[preds_df['pred_id'] == id, 'image_id'].values[0]
        else:
            id = row['target_id']
            image_id = targets_df.loc[targets_df['target_id'] == id, 'image_id'].values[0]

        errors_df.at[index, 'file_name'] = images_df.loc[images_df['image_id'] == image_id, 'file_name'].values[0]
    return errors_df
