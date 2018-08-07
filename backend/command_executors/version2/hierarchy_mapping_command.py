import json
from anytree import Node

from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.common.helper import obtain_dataset_metadata, strcmp, create_dictionary, obtain_dataset_source
from backend.models.musiasem_concepts import Mapping


class HierarchyMappingCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        some_error = False
        issues = []

        # One or more mapping could be specified. The key is "source hierarchy+dest hierarchy"
        # Read mapping parameters

        # Create and store the mapping
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        mappings[self._name] = Mapping(self._name, obtain_dataset_source(origin_dataset), origin_dataset, origin_dimension, destination, map)


        return None, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the metadata dictionary
        return self._content

    def json_deserialize(self, json_input):
        # TODO Check validity
        issues = []
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)

        if "description" in json_input:
            self._description = json_input["description"]
        return issues