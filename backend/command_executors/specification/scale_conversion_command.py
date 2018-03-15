import json
from backend.model_services import IExecutableCommand


class ScaleConversionCommand(IExecutableCommand):
    """
    Useful to convert quantities from one scale to others using a linear transform
    The transform is unidirectional. To define a bidirectional conversion, another scale conversion command is needed
    """
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        """
        """

        return None, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the content
        return self._content

    def json_deserialize(self, json_input):
        # TODO Check validity
        issues = []
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)
        return issues
