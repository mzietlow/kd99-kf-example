from typing import Text, Optional

from tfx import types
from tfx.components.base import executor_spec
from tfx.components.base.base_component import BaseComponent
from tfx.components.base.executor_spec import ExecutorContainerSpec
from tfx.types import standard_artifacts, ComponentSpec, component_spec

from custom_components.file_loader.src import executor


class FileLoaderComponent(BaseComponent):
    class _ComponentSpec(ComponentSpec):
        INPUTS = {
        }
        OUTPUTS = {
        }
        PARAMETERS = {
            'split': component_spec.ExecutionParameter(type=Text),
        }

    SPEC_CLASS = _ComponentSpec

    #  EXECUTOR_SPEC = ExecutorContainerSpec(
    #      image='anylog/file_loader:latest',
    #      command=['python'],
    #      args=['/pipeline/component/src/file_loader.py',
    #            '--split', '{{ exec_props.split }}',
    #            '--output_path', '{{ outputs.data_file.uri }}']
    #  )

    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self, split: str):
        #        data_file = data_file or types.Channel(type=standard_artifacts.ExternalArtifact,
        #                                               artifacts=[standard_artifacts.ExternalArtifact()])
        spec = self._ComponentSpec(split=split)
        super(FileLoaderComponent, self).__init__(spec)
