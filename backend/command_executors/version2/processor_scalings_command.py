import json
from typing import List, Dict, Union

from backend import IssuesOutputPairType, Issue
from backend.model_services import IExecutableCommand, State


class ProcessorScalingsCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content: Dict = {}
        self._description: str = None

    def execute(self, state: State) -> IssuesOutputPairType:
        issues = []

        return issues, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self) -> Dict:
        """Directly return the metadata dictionary"""
        return self._content

    def json_deserialize(self, json_input: Union[dict, str, bytes, bytearray]) -> List[Issue]:
        # TODO Check validity
        issues = []
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)

        if "description" in json_input:
            self._description = json_input["description"]

        return issues
