import os
import subprocess
import tarfile
import time
import configparser
from datetime import datetime

import mlflow
from google.cloud import storage
from googleapiclient import discovery


def make_tarfile(output_filename, source_dir):
    """
    Create .tar.gz file from a directory
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    print(bucket_name)
    print(destination_blob_name)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}/{}.".format(
            bucket_name, source_file_name, destination_blob_name
        )
    )


def package_and_upload_training_code(local_training_dir,
                                     gcs_bucket,
                                     gcs_prefix='training_packages/training_package.tar.gz',
                                     tar_name='training_package.tar.gz'):
    """
    packages training code as a python project and uploads to GCS
    """
    subprocess.run(["python", "setup.py", "sdist"], capture_output=True, cwd=local_training_dir)
    make_tarfile(tar_name, local_training_dir)
    upload_blob(gcs_bucket, os.path.join(os.path.dirname(local_training_dir), tar_name), gcs_prefix)


class GCPTrainingConfig():
    """
    JSON Configuration used to submit jobs to Google ML api
    :param gcs_package_bucket: GCS bucket which contains the modeling script/package
    :type gcs_package_bucket: str
    :param gcs_training_data_path: Full GCS path of training data (i.e. gs://...)
    :type gcs_training_data_path: str
    :param gcs_artifact_bucket: GCS bucket where model artifacts will be stored
    :type gcs_artifact_bucket: str
    :param gcs_package_prefix: GCS prefix of modeling script/package (.tar.gz)
    :type gcs_package_prefix: str
    :param gcs_artifact_prefix: GCS prefix where model artifacts will be stored
    :type gcs_artifact_prefix: str
    :param additional_packages: List of full GCS paths of additional packages to be
        installed for training
    :type additional_packages: list
    :param **kwargs: additional arguments to be passed to TrainingInput. Will be used to
    override any default arguments
    https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#TrainingInput
    """
    def __init__(self,
                 gcs_package_bucket,
                 gcs_training_data_path,
                 gcs_artifact_bucket=None,
                 gcs_package_prefix='training_packages/training_package.tar.gz',
                 gcs_artifact_prefix='artifacts',
                 additional_packages=[],
                 **kwargs):
        self.gcs_package_bucket = gcs_package_bucket
        self.gcs_training_data_path = gcs_training_data_path
        self.gcs_artifact_bucket = gcs_artifact_bucket
        self.gcs_package_prefix = gcs_package_prefix
        self.gcs_artifact_prefix = gcs_artifact_prefix
        self.additional_packages = additional_packages
        self.kwargs = kwargs
        self.base_artifact_path = None
        self.base_config = None
        self.config = None
        self._initialize_config()

    def _initialize_config(self):
        if not self.gcs_artifact_bucket:
            self.gcs_artifact_bucket = self.gcs_package_bucket
        self.base_artifact_path = os.path.join('gs://',
                                               self.gcs_artifact_bucket,
                                               self.gcs_artifact_prefix)
        self.base_config = {
                            'scaleTier': 'CUSTOM',
                            'masterType': 'complex_model_m',
                            'workerType': 'complex_model_m',
                            'parameterServerType': 'large_model',
                            'workerCount': 1,
                            'parameterServerCount': 1,
                            'packageUris': ['gs://{}/{}'.format(self.gcs_package_bucket, self.gcs_package_prefix)]
                            + self.additional_packages,
                            'pythonModule': 'trainer.task',
                            'args': ['--train', self.gcs_training_data_path],
                            'region': 'us-central1',
                            'jobDir': self.base_artifact_path,
                            'runtimeVersion': '2.2',
                            'pythonVersion': '3.7',
                            'scheduling': {'maxWaitTime': '3600s', 'maxRunningTime': '14400s'},
                        }
        self.config = {**self.base_config, **self.kwargs}

    def initialize_output_path(self, full_path=None, suffix=datetime.now().strftime('%Y_%m_%d_%H_%M_%S')):
        if full_path:
            self.config['jobDir'] = full_path
        else:
            self.config['jobDir'] = os.path.join(self.base_artifact_path, suffix)

    def set_hyperparameters(self, metric_name, hyperparams={}, run_tuning=False, tuning_hyperparams={}, **kwargs):
        if run_tuning:
            self.config['hyperparameters'] = tuning_hyperparams
        else:
            self.config['hyperparameters'] = self.process_hyperparameters(hyperparams, metric_name, **kwargs)

    def process_hyperparameters(self, hyperparams, metric_name, **kwargs):
        params = \
            [{'parameterName': param,
              'type': 'CATEGORICAL',
              'categoricalValues': [str(value)]}
             for param, value in hyperparams.items()]
        return {
                'goal': 'MINIMIZE',
                'hyperparameterMetricTag': metric_name,
                'maxTrials': 1,
                'maxParallelTrials': 1,
                'params': params,
                **kwargs}


class GCPTrainingJob():
    """
    Module to submit training jobs to GCP ML API
    """
    def __init__(self,
                 gcp_project_name,
                 job_name,
                 training_config,
                 job_suffix=datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                 version='v1'):
        self.job_name = job_name
        self.gcp_project_name = gcp_project_name
        self.project_id = 'projects/{}'.format(gcp_project_name)
        self.job_id = '{}_{}'.format(job_name,
                                     job_suffix)
        self.tracking_job_id = '{}/jobs/{}'.format(self.project_id, self.job_id)
        self.training_config = training_config
        self.ml = discovery.build('ml', version)
        self.job_submit_response = None
        self.final_job_status_response = None

    def submit_job(self, output_suffix=None):
        output_suffix = output_suffix if output_suffix else self.job_id
        self.training_config.initialize_output_path(suffix=output_suffix)
        job_spec = {'jobId': self.job_id, 'trainingInput': self.training_config.config}
        request = self.ml.projects().jobs().create(body=job_spec,
                                                   parent=self.project_id)
        self.job_submit_response = request.execute()
        return self.job_submit_response

    def track_job(self):
        if self.final_job_status_response:
            return self.final_job_status_response
        while True:
            job_status_response = self.ml.projects().jobs().get(name=self.tracking_job_id).execute()
            state = job_status_response['state']
            print('Time: {}, State: {}'.format(str(datetime.now()), state))
            if state == 'FAILED' or state == 'SUCCEEDED':
                break
            time.sleep(30)
        self.final_job_status_response = job_status_response
        return job_status_response


class GCPMetricsLogger():
    """
    Module to log training job metrics to MLFlow
    """
    def __init__(self, tracking_uri):
        self.tracking_uri = tracking_uri 
    def log_training_metrics(self, training_job, run_name=None, experiment_id=None):
        final_job_status_response = training_job.final_job_status_response
        base_run_name = run_name if run_name else final_job_status_response['jobId']
        base_output_gcs_path = final_job_status_response['trainingInput']['jobDir']
        metric_name = final_job_status_response['trainingInput']['hyperparameters']['hyperparameterMetricTag']
        for trial in final_job_status_response['trainingOutput']['trials']:
            run_name = '{}-trial_{}'.format(base_run_name, trial['trialId'])
            model_gcs_path = os.path.join(base_output_gcs_path, trial['trialId'])
            tags = {'model_gcs_path': model_gcs_path,
                    'packageUris': final_job_status_response['trainingInput']['packageUris'],
                    'args': final_job_status_response['trainingInput']['args']}
            params = trial['hyperparameters']
            metrics = {metric_name: trial['finalMetric']['objectiveValue']}
            self.log_metrics_params_tags(run_name, metrics, params, tags)

    def initialize_mlflow(self, experiment_name):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_metrics_params_tags(self, run_name, metrics=None, params=None, tags=None, experiment_id=None):
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
