import json

from backend.command_executors.analysis.indicators_command import IndicatorsCommand
from backend.command_executors.external_data.mapping_command import MappingCommand
from backend.command_executors.external_data.etl_external_dataset_command import ETLExternalDatasetCommand
from backend.command_executors.external_data.parameters_command import ParametersCommand
from backend.command_executors.specification.dummy_command import DummyCommand
from backend.command_executors.specification.hierarchy_command import HierarchyCommand
from backend.command_executors.specification.metadata_command import MetadataCommand
from backend.command_executors.specification.data_input_command import DataInputCommand
from backend.command_executors.specification.pedigree_matrix_command import PedigreeMatrixCommand
from backend.command_executors.specification.references_command import ReferencesCommand
from backend.command_executors.specification.structure_command import StructureCommand
from backend.command_executors.specification.upscale_command import UpscaleCommand


def create_command(cmd_type, name, json_input):
    """
    Factory creating and initializing a command from its type, optional name and parameters

    :param cmd_type: String describing the type of command, as found in the interchange JSON format
    :param name: An optional name (label) for the command
    :param json_input: Parameters specific of the command type (each command knows how to interpret,
                       validate and integrate them)
    :return: The instance of the command and the issues creating it
    :raise
    """
    cmds = {"dummy":           DummyCommand,  # Simple assignation of a string to a variable. Useful to test things
            "metadata":        MetadataCommand,
            "mapping":         MappingCommand,
            "data_input":      DataInputCommand,
            "structure":       StructureCommand,
            "upscale":         UpscaleCommand,
            "hierarchy":       HierarchyCommand,
            "etl_dataset":     ETLExternalDatasetCommand,
            "parameters":      ParametersCommand,
            "indicators":      IndicatorsCommand,
            "references":      ReferencesCommand,
            "pedigree_matrix": PedigreeMatrixCommand,
            }
    if cmd_type in cmds:
        tmp = cmds[cmd_type](name)  # Reflective CALL to construct the empty command instance
        tmp._serialization_type = cmd_type  # Injected attribute. Used later for serialization
        tmp._serialization_label = name  # Injected attribute. Used later for serialization
        if isinstance(json_input, (str, dict, list)):
            if json_input != {}:
                issues = tmp.json_deserialize(json_input)
            else:
                issues = []
        else:
            # NO SPECIFICATION
            raise Exception("The command '" + cmd_type + " " + name if name else "<unnamed>" + " does not have a specification.")
        return tmp, issues  # Return the command and the issues found during the deserialization
    else:
        # UNKNOWN COMMAND
        raise Exception("Unknown command type: " + cmd_type)


def commands_generator_from_native_json(input, state):
    """
    It allows both a single command ("input" is a dict) or a sequence of commands ("input" is a list)

    :param input: Either a dict (for a single command) or list (for a sequence of commands)
    :return: A generator of IExecutableCommand
    """
    def build_and_yield(d):
        # Only DICT is allowed, and the two mandatory attributes, "command" and "content"
        if isinstance(d, dict) and "command" in d and "content" in d:
            if "label" in d:
                n = d["label"]
            else:
                n = None
            yield create_command(d["command"], n, d["content"])

    j = json.loads(input)  # Convert JSON string to dictionary
    if isinstance(j, list):  # A sequence of primitive commands
        for i in j:  # For each member of the list
            yield from build_and_yield(i)
    else:  # A single primitive command
        yield from build_and_yield(j)
