import argparse
import logging
import openml
import os


def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default=os.path.expanduser('~/experiments/metadl'), type=str)
    args_, misc = parser.parse_known_args()

    return args_


def run(args):
    num_episodes = 100
    all_dids = []
    all_molecules = set()
    for dataset_id in range(23520, 40462):
        try:
            valid = True
            data = openml.datasets.get_dataset(dataset_id)
            if data.default_target_attribute != 'pXC50':
                # logging.warning('target does not match %d' % dataset_id)
                continue
            # logging.info('obtained dataset %d' % dataset_id)

            frame, _, _, columns = data.get_data(include_row_id=True, target=data.default_target_attribute)
            if len(columns) != 1025:
                continue
            if frame.shape[0] < 80 or frame.shape[0] > 300:
                continue

            for column in columns:
                if column != 'molecule_id' and column != 'pXC50' and not column.startswith('FCFP4_1024'):
                    # logging.warning('attributes do not match %d' % dataset_id)
                    valid = False
            molecules = set(frame['molecule_id'].unique())
            for molecule in molecules:
                if molecule in all_molecules:
                    valid = False
            if valid:
                all_molecules.update(molecules)
                all_dids.append(dataset_id)
                logging.info("(%d/%d) did %d shape: %s" % (len(all_dids), num_episodes, dataset_id, str(frame.shape)))
            if len(all_dids) == num_episodes:
                print(all_dids)
                all_dids = []
        except ValueError:
            logging.error('problem with dataset %d' % dataset_id)
        except openml.exceptions.OpenMLServerException:
            logging.error('problem with dataset %d' % dataset_id)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    run(read_cmd())
