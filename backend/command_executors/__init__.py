from typing import Optional

from backend import ExecutableCommandIssuesPairType, Command
from backend.command_definitions import commands
from backend.common.helper import first


def create_command(cmd_type, name, json_input, source_block=None) -> ExecutableCommandIssuesPairType:
    """
    Factory creating and initializing a command from its type, optional name and parameters

    :param cmd_type: String describing the type of command, as found in the interchange JSON format
    :param name: An optional name (label) for the command
    :param json_input: Parameters specific of the command type (each command knows how to interpret,
                       validate and integrate them)
    :param source_block: String defining the name of the origin block (in a spreadsheet is the worksheet name, but other block types could appear)
    :return: The instance of the command and the issues creating it
    :raise
    """
    cmd: Optional[Command] = first(commands, condition=lambda c: c.name == cmd_type)

    if cmd:
        exec_cmd: "IExecutableCommand" = cmd.execution_class(name)  # Reflective call
        exec_cmd._serialization_type = cmd_type  # Injected attribute. Used later for serialization
        exec_cmd._serialization_label = name  # Injected attribute. Used later for serialization
        exec_cmd._source_block_name = source_block
        if isinstance(json_input, (str, dict, list)):
            if json_input != {}:
                issues = exec_cmd.json_deserialize(json_input)
            else:
                issues = []
        else:
            # NO SPECIFICATION
            raise Exception("The command '" + cmd_type + " " + name if name else "<unnamed>" + " does not have a specification.")
        return exec_cmd, issues  # Return the command and the issues found during the deserialization
    else:
        # UNKNOWN COMMAND
        raise Exception("Unknown command type: " + cmd_type)
