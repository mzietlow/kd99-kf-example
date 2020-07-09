from typing import Dict, Text, List, Any

from tfx import types
from tfx.components.base import base_executor


class Executor(base_executor.BaseExecutor):
    """DemoExecutor for HelloComponent."""

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        print("Arrived in DemoExecutor!!!")
        split_to_instance = {}
