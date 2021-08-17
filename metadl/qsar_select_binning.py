import argparse
import numpy as np
import logging
import os
import pandas as pd
import sklearn
import sklearn.ensemble

import openml.datasets


def read_cmd():
    dataset_1 = \
        [23545, 23588, 23621, 23644, 23652, 23659, 23701, 23824, 23971, 23975,
         23995, 24010, 24041, 24091, 24156, 24166, 24186, 24208, 24284, 24286,
         24321, 24348, 24385, 24476, 24605, 24655, 24673, 24713, 24761, 24803,
         24854, 24903, 24910, 24957, 25003, 25012, 25039, 25084, 25112, 25160,
         25162, 25340, 25344, 25380, 25393, 25463, 25481, 25570, 25620, 25703,
         25752, 25773, 25810, 25847, 25889, 25960, 26254, 26267, 26278, 26340,
         26342, 26370, 26388, 26414, 26424, 26540, 26550, 26577, 26812, 26886,
         26948, 26952, 26965, 27031, 27037, 27186, 27287, 27325, 27374, 27416,
         27545, 27768, 27771, 27804, 27941, 27991, 27996, 28112, 28120, 28258,
         28304, 28399, 28418, 28666, 28718, 28756, 28824, 28894, 28918, 28946,
         28961, 28978, 29002, 29103, 29136, 29138, 29409, 29441, 29602, 29692,
         29705, 29736, 29825, 29872, 29874, 29958, 30044, 30180, 30214, 30255,
         30525, 30572, 30584, 30794, 30824, 30897, 30929, 30963, 30995, 31007,
         31126, 31252, 31274, 31323, 31379, 31418, 31551, 31968, 32028, 32150,
         32162, 32187, 32206, 32234, 32315, 32334, 32400, 32515, 32521, 32686,
         32699, 32713, 32935, 33035, 33110, 33128, 33240, 33309, 33381, 33463,
         33732, 33790, 33902, 33909, 33928]
    dataset_2 = \
        [33966, 34024, 34035, 34359, 34557,
         34613, 34687, 34733, 35308, 35359, 35423, 35738, 35822, 35844, 35871,
         36295, 36663, 36688, 36916, 36923, 36935, 37036, 37165, 37253, 37534,
         37571, 37584, 37769, 37869, 37922, 37973, 38207, 38662, 38669, 38678]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_ids', nargs="+", default=dataset_1, type=int)
    parser.add_argument('--min_examples', default=10, type=int)
    parser.add_argument('--output_dir', default=os.path.expanduser('~/experiments/metadl/'), type=str)
    args_, misc = parser.parse_known_args()

    return args_


def rename_col(name: str):
    if name.startswith('FCFP4_1024b'):
        name = 'Feat' + name[11:]
    if name == 'pXC50':
        name = 'CATEGORY'
    return name


def format_frame(frame: pd.DataFrame, did: int):
    frame['pXC50'] = frame['pXC50'].replace(0, 'vlow')
    frame['pXC50'] = frame['pXC50'].replace(2, 'low')
    frame['pXC50'] = frame['pXC50'].replace(4, 'medium')
    frame['pXC50'] = frame['pXC50'].replace(6, 'high')
    frame['pXC50'] = frame['pXC50'].replace(8, 'vhigh')
    frame['SUPER_CATEGORY'] = did
    frame = frame.rename(columns=lambda l: rename_col(l))

    return frame


def run_classifier_on_frame(frame):
    y = frame['pXC50'].values
    del frame['pXC50']
    X = frame.to_numpy(dtype=float)
    clf = sklearn.ensemble.RandomForestClassifier(random_state=42)
    scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=5)
    return np.mean(scores)


def run(args):
    all_frames = []
    for did in args.dataset_ids:
        data = openml.datasets.get_dataset(did)
        frame, _, _, _ = data.get_data()
        classes = frame['pXC50'].to_numpy().reshape(-1, 1)
        # binning is not perfect, due to duplicate values
        discretizer = sklearn.preprocessing.KBinsDiscretizer(n_bins=9, encode='ordinal', strategy='quantile')
        discretizer.fit(classes)
        res = discretizer.transform(classes).reshape((-1, ))
        unique, counts = np.unique(res, return_counts=True)
        if len(counts) < 9 or counts[0] < args.min_examples or counts[2] < args.min_examples or counts[4] < args.min_examples or counts[6] < args.min_examples or counts[8] < args.min_examples:
            continue
        frame['pXC50'] = np.array(res, dtype=int)
        frame = frame[frame.pXC50 % 2 == 0]
        mean_score = run_classifier_on_frame(frame.copy(deep=True))
        if mean_score < 0.2:
            logging.info('Mean score of dataset %d low: %f' (did, mean_score))
        frame = format_frame(frame, did)
        all_frames.append(frame)

    total_frame = pd.concat(all_frames)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = args.output_dir + '/qsar1.csv'
    total_frame.to_csv(output_file)
    logging.info('saved to %s' % output_file)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    run(read_cmd())
