import json
from backend.model_services import IExecutableCommand


class PedigreeMatrixCommand(IExecutableCommand):
    """
    Declaration of a pedigree matrix, which can be used

    """
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        """
        Create the PedigreeMatrix object to which QQ observation may refer

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
