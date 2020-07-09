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
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from custom_components.tfx_demo_component.src import demo_component

_pipeline_name = 'demo-pipe'

_persistent_volume_claim = 'tfx-pvc'
_persistent_volume = 'tfx-pv'
_persistent_volume_mount = '/mnt'

_input_base = os.path.join(_persistent_volume_mount, 'kddcup')
_output_base = os.path.join(_persistent_volume_mount, 'pipelines')
_tfx_root = os.path.join(_output_base, 'tfx')
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)

_data_root = os.path.join(_input_base, 'data')
chicago_taxi_pipeline_root = "/tfx-src/tfx/examples/chicago_taxi_pipeline/data/simple/"

_module_file = os.path.join(_input_base, 'utils.py')

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
    file_loader = demo_component.FileLoaderComponent(split='train')

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            file_loader
        ],
        beam_pipeline_args=beam_pipeline_args)


def mount_existing_pvc(pvc_name='pipeline-claim', volume_name='pipeline', volume_mount_path='/mnt/pipeline',
                       host_path: str = None):
    """
        Modifier function to apply to a Container Op to simplify volume, volume mount addition and
        enable better reuse of volumes, volume claims across container ops.
        Usage:
            train = train_op(...)
            train.apply(mount_pvc('claim-name', 'pipeline', '/mnt/pipeline'))
    """
    if host_path is None:
        return onprem.mount_pvc(pvc_name, volume_name, volume_mount_path)
    else:
        def _mount_pvc(task):
            from kubernetes import client as k8s_client
            return (
                task
                    .add_volume(
                    k8s_client.V1Volume(name=volume_name,
                                        host_path=k8s_client.V1HostPathVolumeSource(path=host_path))
                )
                    .add_volume_mount(
                    k8s_client.V1VolumeMount(mount_path=volume_mount_path, name=volume_name)
                )
            )

        return _mount_pvc


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
    tfx_image = "anylog/demo_component:v0.1"
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
