import json

from backend.model.memory.musiasem_concepts import Parameter
from backend.model_services import IExecutableCommand, get_case_study_registry_objects


class ParametersCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        # Obtain global variables in state
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)

        for param in self._content:
            name = param["name"]
            p = Parameter(name)
            if "value" in param:
                p._default_value = p._current_value = param["value"]
            if "type" in param:
                p._type = param["type"]
            if "range" in param:
                p._range = param["range"]
            if "description" in param:
                p._description = param["description"]
            if "group" in param:
                p._group = None
            glb_idx.put(p.key(), p)
        return None, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the metadata dictionary
        return self._content

    def json_deserialize(self, json_input):
        # TODO Check validity
        issues = []
        if isinstance(json_input, (dict, list)):
            self._content = json_input
        else:
            self._content = json.loads(json_input)
        return issues