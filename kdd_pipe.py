# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Chicago Taxi example using TFX DSL on Kubeflow (runs locally on cluster).
Source: https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_kubeflow_local.py
"""

import os
from typing import List, Text

from kfp import onprem
from tfx.components import CsvExampleGen, ExampleValidator, Transform, Trainer
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input

_pipeline_name = 'kdd-pipe'

_persistent_volume_claim = 'tfx-pvc'
_persistent_volume = 'tfx-pv'
_persistent_volume_mount = '/mnt'

_input_base = os.path.join(_persistent_volume_mount, 'kddcup')
_output_base = os.path.join(_persistent_volume_mount, 'pipelines')
_tfx_root = os.path.join(_output_base, 'tfx')
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)

_tfx_root = "/tfx-src"

_data_root = os.path.join(_tfx_root, 'data/train.small')

_module_file = os.path.join(_tfx_root, 'tfx_utils.py')

_serving_model_dir = os.path.join(_output_base, _pipeline_name, 'serving_model')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     module_file: Text, serving_model_dir: Text,
                     beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
    """Implements the chicago taxi pipeline with TFX and Kubeflow Pipelines."""
    examples = external_input(data_root)

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = CsvExampleGen(input=examples)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file)

    # Uses user-provided Python function that implements a model using TF-Learn
    # to train a model on Google Cloud AI Platform.
    trainer = Trainer(
        module_file=module_file,
        transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000),
    )
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform
        ],
        beam_pipeline_args=beam_pipeline_args)


if __name__ == '__main__':
    # Metadata config. The defaults work with the installation of
    # KF Pipelines using Kubeflow. If installing KF Pipelines using the
    # lightweight deployment option, you may need to override the defaults.
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    # Workaround for TFX issues on KF1.0, see github.com/tensorflow/tfx/issues/1287#issuecomment-598683909
    metadata_config.mysql_db_service_host.value = 'mysql.kubeflow'
    metadata_config.mysql_db_service_port.value = "3306"
    metadata_config.mysql_db_name.value = "metadb"
    metadata_config.mysql_db_user.value = "root"
    metadata_config.mysql_db_password.value = ""
    metadata_config.grpc_config.grpc_service_host.value = 'metadata-grpc-service'
    metadata_config.grpc_config.grpc_service_port.value = '8080'

    # This pipeline automatically injects the Kubeflow TFX image if the
    # environment variable 'KUBEFLOW_TFX_IMAGE' is defined. Currently, the tfx
    # cli tool exports the environment variable to pass to the pipelines.
    tfx_image = "anylog/kdd_data:v0.1"
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        # Specify custom docker image to use.
        tfx_image=tfx_image,
        pipeline_operator_funcs=(
            # If running on K8s Engine (GKE) on Google Cloud Platform (GCP),
            # kubeflow_dag_runner.get_default_pipeline_operator_funcs() provides
            # default configurations specifically for GKE on GCP, such as secrets.
            [
                onprem.mount_pvc(_persistent_volume_claim, _persistent_volume,
                                 _persistent_volume_mount)
            ]))

    kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            data_root=_data_root,
            module_file=_module_file,
            serving_model_dir=_serving_model_dir,
            beam_pipeline_args=_beam_pipeline_args))
