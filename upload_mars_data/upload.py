
import argparse
import logging

import numpy as np
import openml
import os
import pandas as pd

from openml.datasets.functions import create_dataset



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, default='https://test.openml.org/api/v1/xml/')
    parser.add_argument('--apikey', type=str, default='48830dd663e41d5cb689016a072e6ec1')
    parser.add_argument('--data_file', type=str, default='data60.csv')
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.info('Started %s' % os.path.basename(__file__))

    if args.apikey is None:
        print('Api key:')
        args.apikey = input()

    openml.config.server = args.server
    openml.config.apikey = args.apikey
    df = pd.read_csv(args.data_file)
    df = df.replace('?', np.nan)

    df = df.head(100)


    for idx, column in enumerate(df.columns):
        df[column] = df[column].astype(float)
        # print(idx, df.columns.values[idx], df[column].min, 1.0)

    description = 'The collection consists of six data sets containing telemetry data of the Mars Express Spacecraft (MEX), a spacecraft orbiting Mars and operated by the European Space Agency. The data, in terms of context data and thermal power consumption measurements, capture the status of the spacecraft over four Martian years sampled at six different time resolutions ranging from 1 min to 60 min. From a data analysis point-of-view, analysing these data presents great challenges - even for the more sophisticated state-of-the-art artificial intelligence methods. In particular, given the heterogeneity, complexity and magnitude of the data, they can be employed in different scenarios and analysed through the prism of a variety of machine learning tasks, such as multi-target regression, learning from data streams, anomaly detection, clustering etc. While analysing MEX"s telemetry data is critical for aiding very important decisions regarding the spacecraft status, it can be used to extract novel knowledge and monitor the spacecrafts" health, but also to benchmark artificial intelligence methods designed for a variety of tasks.'
    creator = 'Džeroski, Sašo; Ženko, Bernard; Simidjievski, Nikola; Breskvar, Martin; Petkovic, Matej; Kocev, Dragi; et al. (2022)'

    mars_dataset = create_dataset(
        # The name of the dataset (needs to be unique).
        # Must not be longer than 128 characters and only contain
        # a-z, A-Z, 0-9 and the following special characters: _\-\.(),
        name="ThermalPowerConsumptionMarsExpress60",
        # Textual description of the dataset.
        description="TODO",        # The person who created the dataset.
        creator="todo",
        # People who contributed to the current version of the dataset.
        contributor=None,
        # The date the data was originally collected, given by the uploader.
        collection_date="09-01-2012",
        # Language in which the data is represented.
        # Starts with 1 upper case letter, rest lower case, e.g. 'English'.
        language="English",
        # License under which the data is/will be distributed.
        licence="Unknown",
        # Name of the target. Can also have multiple values (comma-separated).
        default_target_attribute=None,
        # The attribute that represents the row-id column, if present in the
        # dataset.
        row_id_attribute=None,
        # Attribute or list of attributes that should be excluded in modelling, such as
        # identifiers and indexes. E.g. "feat1" or ["feat1","feat2"]
        ignore_attribute=None,
        # How to cite the paper.
        citation="TODO",
        attributes="auto",
        data=df,
        # A version label which is provided by the user.
        version_label="test",
        original_data_url="https://springernature.figshare.com/collections/Machine-learning_ready_data_on_the_Thermal_Power_Consumption_of_the_Mars_Express_Spacecraft/5360420/1",
        paper_url="https://www.nature.com/articles/s41597-022-01336-z",
    )
    mars_dataset.publish()

    print(f"URL for dataset: {mars_dataset.openml_url}")


if __name__ == '__main__':
    run(parse_args())
