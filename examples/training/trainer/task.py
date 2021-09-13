from __future__ import print_function


from io import StringIO
import os
import sys
import subprocess

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

import hypertune


def log_normalize_features(data_df, method='skewed_only', skew_threshold=1):
    if not isinstance(data_df, pd.DataFrame):
        data_df = pd.DataFrame(data_df)
    if method == 'none':
        return data_df
    elif method == 'skewed_only':
        skewness = stats.skew(data_df)
        log_data_df = data_df.copy()
        for col, skew in zip(data_df.columns, skewness):
            if abs(skew) >= skew_threshold:
                log_data_df[col] = np.log(data_df[col])
        return log_data_df
    elif method == 'all':
        return pd.DataFrame(np.log(data_df), columns=data_df.columns, index=data_df.index)
    else:
        raise ValueError('Unknown normalize method {}'.format(method))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #Job Parameters
    parser.add_argument('--standardize', type=bool, default=True)
    parser.add_argument('--normalize', type=str, default='skewed_only')
    parser.add_argument('--skew_threshold', type=float, default=1.0)
    parser.add_argument('--reduce', type=bool, default=False)
    parser.add_argument('--num_dimensions', type=int, default=50)
    parser.add_argument('--num_clusters', type=int, required=True)


    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--job-dir', type=str)
    parser.add_argument('--model-dir', type=str, default="./")
    parser.add_argument('--train', type=str, help='GCS storage directory path', required=True)

    args = parser.parse_args()
    local_dir = os.path.basename(args.train)
    if args.train[-1] == '/':
        local_dir = os.path.basename(args.train[:-1])
    subprocess.check_call(['gsutil', 'cp', '-r', args.train, './'], stderr=sys.stdout)
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [os.path.join(args.train, file) for file in os.listdir(local_dir)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [pd.read_csv(file, header=None) for file in input_files]
    concat_data = pd.concat(raw_data)

    scaler = StandardScaler()
    normalizer = FunctionTransformer(log_normalize_features,
                                     kw_args={'method': args.normalize,
                                              'skew_threshold': args.skew_threshold})

    steps = [('normalize', normalizer), ('scale', scaler)]
    if args.reduce:
        num_dimensions = min(args.num_dimensions, concat_data.shape[1])
        reducer = PCA(n_components=num_dimensions)
        steps.append(('reduce', reducer))

    kmeans = KMeans(n_clusters=args.num_clusters)
    steps.append(('cluster', kmeans))

    pipeline = Pipeline(steps)
    pipeline.fit(concat_data)
    ssd = pipeline['cluster'].inertia_
    print('Training:SSD = {};'.format(ssd))
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
                                            hyperparameter_metric_tag='ssd',
                                            metric_value=ssd)
    joblib.dump(pipeline, os.path.join(args.model_dir, "model.joblib"))
    subprocess.check_call(['gsutil', 'cp', 'model.joblib', args.job_dir], stderr=sys.stdout)
    print("saved model!")
