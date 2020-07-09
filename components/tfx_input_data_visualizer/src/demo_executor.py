from typing import Dict, Text, List, Any

from tfx import types
from tfx.components.base import base_executor


class DemoExecutor(base_executor.BaseExecutor):
    """DemoExecutor for HelloComponent."""

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:

        split_to_instance = {}
