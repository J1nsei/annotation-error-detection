import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import Callable, Dict, Tuple
import pandas as pd
from tqdm import tqdm
from utils.find_errors import ErrorType, TARGETS_DF_COLUMNS, PREDS_DF_COLUMNS


class MyMeanAveragePrecision:

    def __init__(self, foreground_threshold):
        self.device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.map = MeanAveragePrecision(
            iou_thresholds=[foreground_threshold]
        ).to(self.device)

    def __call__(self, targets_df, preds_df, error_type):
        targets, preds = self._format_inputs(targets_df, preds_df, error_type)
        self.map.update(preds=preds, target=targets)
        result = self.map.compute()["map"].item()
        self.map.reset()
        return result

    def _format_inputs(self, targets_df, preds_df, error_type):
        image_ids = set(targets_df["image_id"]) | set(preds_df["image_id"])
        targets, preds = [], []
        desc = 'Calculating ' + error_type + ' impact'
        for image_id in tqdm(image_ids, desc=desc):
            im_targets_df = targets_df.query("image_id == @image_id")
            im_preds_df = preds_df.query("image_id == @image_id")
            targets.append(
                {
                    "boxes": torch.as_tensor(
                        im_targets_df[["xmin", "ymin", "xmax", "ymax"]].values,
                        dtype=torch.float32,
                    ).to(self.device),
                    "labels": torch.as_tensor(
                        im_targets_df["label_id"].values, dtype=torch.int64
                    ).to(self.device),
                }
            )
            preds.append(
                {
                    "boxes": torch.as_tensor(
                        im_preds_df[["xmin", "ymin", "xmax", "ymax"]].values,
                        dtype=torch.float32,
                    ).to(self.device),
                    "labels": torch.as_tensor(
                        im_preds_df["label_id"].to_numpy(dtype='int16'), dtype=torch.int64
                    ).to(self.device),
                    "scores": torch.as_tensor(
                        im_preds_df["score"].values, dtype=torch.float32
                    ).to(self.device),
                }
            )
        return targets, preds


def calculate_error_impact(
        metric_name: str,
        metric: Callable,
        errors_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        preds_df: pd.DataFrame,
) -> Dict[str, float]:

    print('\nCalculating the impact of errors \n')
    ensure_consistency(errors_df, targets_df, preds_df)

    metric_values = {
        ErrorType.CLS: metric(*fix_cls_error(errors_df, targets_df, preds_df), ErrorType.CLS),
        ErrorType.LOC: metric(*fix_loc_error(errors_df, targets_df, preds_df), ErrorType.LOC),
        ErrorType.CLS_LOC: metric(
            *fix_cls_loc_error(errors_df, targets_df, preds_df), ErrorType.CLS_LOC
        ),
        ErrorType.DUP: metric(*fix_dup_error(errors_df, targets_df, preds_df), ErrorType.DUP),
        ErrorType.BKG: metric(*fix_bkg_error(errors_df, targets_df, preds_df), ErrorType.BKG),
        ErrorType.MISS: metric(
            *fix_miss_error(errors_df, targets_df, preds_df), ErrorType.MISS
        ),
    }


    baseline_metric = metric(targets_df, preds_df, 'baseline')
    impact = {
        error: (error_metric - baseline_metric)
        for error, error_metric in metric_values.items()
    }
    impact[metric_name] = baseline_metric

    return impact


def ensure_consistency(
        errors_df: pd.DataFrame, targets_df: pd.DataFrame, preds_df: pd.DataFrame
):

    target_ids = set(targets_df["target_id"])
    pred_ids = set(preds_df["pred_id"])

    error_target_ids = set(errors_df.query("target_id.notnull()")["target_id"])
    error_pred_ids = set(errors_df.query("pred_id.notnull()")["pred_id"])

    if not target_ids == error_target_ids:
        raise ValueError(
            f"Missing target IDs in error_df: {target_ids - error_target_ids}"
        )

    if not pred_ids == error_pred_ids:
        raise ValueError(
            f"Missing pred IDs in error_df: {pred_ids - error_pred_ids}"
        )


def fix_cls_error(
        errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _fix_by_correcting_and_removing_preds(
        errors_df, targets_df, preds_df, ErrorType.CLS
    )


def fix_loc_error(
        errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _fix_by_correcting_and_removing_preds(
        errors_df, targets_df, preds_df, ErrorType.LOC
    )


def fix_cls_loc_error(
        errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _fix_by_removing_preds(
        errors_df, targets_df, preds_df, ErrorType.CLS_LOC
    )


def fix_bkg_error(
        errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _fix_by_removing_preds(
        errors_df, targets_df, preds_df, ErrorType.BKG
    )


def fix_dup_error(
        errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _fix_by_removing_preds(
        errors_df, targets_df, preds_df, ErrorType.DUP
    )


def fix_miss_error(
        errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    ensure_consistency(errors_df, targets_df, preds_df)
    miss = ErrorType.MISS
    targets_df = targets_df.merge(

        errors_df.query("error_type == @miss"),
        on="target_id",
        how="left",
    ).query("error_type.isnull()")
    return targets_df[TARGETS_DF_COLUMNS], preds_df


def _fix_by_correcting_and_removing_preds(
        errors_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        preds_df: pd.DataFrame,
        error_type: ErrorType,
) -> Tuple[pd.DataFrame, pd.DataFrame]:


    assert error_type in {
        ErrorType.CLS,
        ErrorType.LOC,
    }, f"error_type='{error_type}'"
    ensure_consistency(errors_df, targets_df, preds_df)

    cols_to_correct = {
        ErrorType.CLS: ["label_id"],
        ErrorType.LOC: ["xmin", "ymin", "xmax", "ymax"],
    }[error_type]
    error_types = [ErrorType.OK, ErrorType.CLS, ErrorType.LOC]

    preds_df = (
        preds_df.merge(
            errors_df.query(
                "error_type in @error_types"
            ),
            on="pred_id",
            how="left",
        )
        .merge(
            targets_df[["target_id"] + cols_to_correct],
            on="target_id",
            how="left",
            suffixes=("", "_target"),
        )
        .sort_values(by="score", ascending=False)
    )

    to_correct = preds_df["error_type"].eq(error_type)
    target_cols = [col + "_target" for col in cols_to_correct]
    preds_df.loc[to_correct, cols_to_correct] = preds_df.loc[
        to_correct, target_cols
    ].values

    to_drop = []
    desc = 'Fixing ' + error_type + ' error'
    for _, target_df in tqdm(preds_df.groupby("target_id"), desc=desc):
        if target_df["error_type"].eq(ErrorType.OK).any():

            to_drop += target_df.query("error_type == @error_type")[
                "pred_id"
            ].tolist()
        elif (
                target_df["error_type"].eq(error_type).any() and len(target_df) > 1
        ):

            to_keep = target_df["pred_id"].iloc[0]
            to_drop += target_df.query(
                "error_type == @error_type and pred_id != @to_keep"
            )["pred_id"].tolist()
    return (
        targets_df,
        preds_df.query("pred_id not in @to_drop")[PREDS_DF_COLUMNS],
    )


def _fix_by_removing_preds(
        errors_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        preds_df: pd.DataFrame,
        error_type: ErrorType,
) -> Tuple[pd.DataFrame, pd.DataFrame]:


    assert error_type in {
        ErrorType.CLS_LOC,
        ErrorType.BKG,
        ErrorType.DUP,
    }, f"error_type='{error_type}'"
    ensure_consistency(errors_df, targets_df, preds_df)

    preds_df = preds_df.merge(errors_df, on="pred_id", how="left").query(
        "error_type != @error_type"
    )
    return targets_df, preds_df[PREDS_DF_COLUMNS]
