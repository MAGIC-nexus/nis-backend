import json
from backend.model_services import IExecutableCommand


class ScaleConversionCommand(IExecutableCommand):
    """
    Useful to convert quantities from one scale to others using a linear transform
    The transform is unidirectional. To define a bidirectional conversion, another scale conversion command is needed
    """
    def __init__(self, name: str):
        self._name = name

    def execute(self, state: "State"):
        """ The execution creates an instance of a Metadata object, and assigns the name "metadata" to the variable,
            inserting it into "State" 
        """
        state.set(self._var_name, self._description)
        return None, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the metadata dictionary
        return {"name": self._var_name, "description": self._description}

    def json_deserialize(self, json_input):
        # TODO Read and check keys validity
        issues = []
        if isinstance(json_input, dict):
            pass
        else:
            json_input = json.loads(json_input)

        if "name" in json_input:
            self._var_name = json_input["name"]
        if "description" in json_input:
            self._description = json_input["description"]

        return issues