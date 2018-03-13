import json
from backend.model_services import IExecutableCommand, get_case_study_registry_objects


class ReferencesCommand(IExecutableCommand):
    """ It is a mere data container
        Depending on the type, the format can be controlled with a predefined schema
    """
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        """
        Process each of the references, simply storing them as Reference objects
        """
        glb_idx, _, _, _, _ = get_case_study_registry_objects(state)

        # TODO Process each reference
        for i, o in enumerate(self._content):
            pass

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
