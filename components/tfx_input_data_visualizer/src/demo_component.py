from typing import Text

from tfx.components.base import executor_spec
from tfx.components.base.base_component import BaseComponent
from tfx.types import ComponentSpec, component_spec

from custom_components.tfx_demo_component.src import demo_executor


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

    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(demo_executor.DemoExecutor)

    def __init__(self, split: str):
        spec = self._ComponentSpec(split=split)
        super(FileLoaderComponent, self).__init__(spec)
