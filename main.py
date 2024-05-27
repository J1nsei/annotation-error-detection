from ultralytics import YOLO
from utils.utils import *
from utils.dataset_creation import *
from utils.find_errors import *
from utils.visualization import *
from utils.evaluate_errors import *
import os
import yaml
import warnings
warnings.filterwarnings('ignore')

def main():
    args = main_arg_parser()
    DATA_PATH = Path(args.data)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data path '{DATA_PATH}' does not exist.")

    MODEL = args.model
    TRAIN_MODE = args.train.lower() == 'true'
    N_SPLITS = 5
    CREATE_DATASET = args.create_dataset.lower() == 'true'
    TRAIN_FRACTION = args.trainsz
    IMPACT = args.impact.lower() == 'true'
    PREDS = args.preds
    SAVE_PREDS = args.save_preds.lower() == 'true'
    VIS = args.vis
    SAVE_VDATA = args.save_vdata.lower() == 'true'
    LABELS = args.labels
    OWN_IMPACT = args.own_impact
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL)




    if TRAIN_MODE:
        with open('models/train_config.yaml', 'r') as train_cfg:
            train_args = yaml.safe_load(train_cfg)
        create_yolo_dataset(DATA_PATH, TRAIN_FRACTION, N_SPLITS) if CREATE_DATASET else None
        train_args['project'] = 'trained_yolo'
        for i in range(N_SPLITS):
            print(f'Training split #{i + 1} of {N_SPLITS}:\n')
            train_args['data'] = f'data_{i + 1}.yaml'
            train_args['name'] = f'split_{i + 1}'
            model.train(device=device, **train_args)

    else:
        with open('models/pred_config.yaml', 'r') as pred_cfg:
            pred_args = yaml.safe_load(pred_cfg)
        images_df, targets_df = load_dataset(DATA_PATH, LABELS)
        images_path, id2label = get_utils_variables(DATA_PATH, LABELS)
        preds_df = get_predictions(model, device, images_path, images_df, id2label, pred_args, PREDS)
        errors_df = classify_predictions_errors(targets_df, preds_df, images_df)
        os.makedirs('errors_found', exist_ok=True)
        os.makedirs('predictions', exist_ok=True)
        print(errors_df.columns)
        errors_df.to_pickle(f'errors_found/errors_df_{LABELS}.pkl', )
        preds_df.to_pickle(f'./predictions/preds_{LABELS}.pkl') if SAVE_PREDS else None
        print(errors_df["error_type"].value_counts())
        if IMPACT:
            if OWN_IMPACT:
                with open("impact.json", "r") as f:
                    impact = json.load(f)
            else:
                impact = calculate_error_impact(
                    "mAP@50",
                    MyMeanAveragePrecision(foreground_threshold=FOREGROUND_IOU_THRESHOLD),
                    errors_df,
                    targets_df,
                    preds_df
                )
                with open("impact.json", "w") as f:
                    json.dump(impact, f)
            make_charts(impact)
        visualize(DATA_PATH, images_path, targets_df, images_df, preds_df, errors_df, id2label, LABELS, VIS, SAVE_VDATA)
    print('\nDONE')


if __name__ == "__main__":
    main()
