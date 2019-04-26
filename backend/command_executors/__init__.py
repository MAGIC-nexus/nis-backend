import json
from typing import Optional, List, Dict, Any, Union, NoReturn

from backend import ExecutableCommandIssuesPairType, Command, CommandField, IssuesOutputPairType
from backend.command_definitions import commands
from backend.common.helper import first, PartialRetrievalDictionary, head
from command_generators import IType, IssueLocation, Issue
from command_generators.parser_ast_evaluators import dictionary_from_key_value_list
from model_services import IExecutableCommand, get_case_study_registry_objects
from models.musiasem_concepts import Processor, Factor
from models.musiasem_concepts_helper import find_processor_by_name


class CommandExecutionError(Exception):
    pass


class BasicCommand(IExecutableCommand):
    def __init__(self, name: str, command_fields: List[CommandField]):
        self._name = name
        self._content: Dict = {}
        self._command_name = ""
        self._command_fields = command_fields

        # Execution state
        self._issues: List[Issue] = None
        self._current_row_number: int = None
        self._glb_idx: PartialRetrievalDictionary = None
        self._fields_values = {}

    def _get_command_fields_values(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {f.name: row.get(f.name, head(f.allowed_values)) for f in self._command_fields}

    def _check_all_mandatory_fields_have_values(self) -> NoReturn:
        empty_fields: List[str] = [f.name
                                   for f in self._command_fields
                                   if f.mandatory and self._fields_values[f.name] is None]

        if len(empty_fields) > 0:
            raise CommandExecutionError(f"Mandatory field/s '{', '.join(empty_fields)}' is/are empty.")

    def _add_issue(self, itype: IType, description: str) -> NoReturn:
        self._issues.append(
            Issue(itype=itype,
                  description=description,
                  location=IssueLocation(sheet_name=self._command_name,
                                         row=self._current_row_number, column=None))
        )
        return

    def estimate_execution_time(self) -> int:
        return 0

    def json_serialize(self) -> Dict:
        """Directly return the metadata dictionary"""
        return self._content

    def json_deserialize(self, json_input: Union[dict, str, bytes, bytearray]) -> List[Issue]:
        # TODO Check validity
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)

        self._command_name = self._content["command_name"]

        return []

    def execute(self, state: "State") -> IssuesOutputPairType:
        self._issues = []

        self._glb_idx, _, _, _, _ = get_case_study_registry_objects(state)

        for row in self._content["items"]:
            try:
                self._init_and_process_row(row)
            except CommandExecutionError as e:
                self._add_issue(IType.ERROR, str(e))

        return self._issues, None

    def _init_and_process_row(self, row: Dict[str, Any]) -> NoReturn:
        self._current_row_number = row["_row"]
        self._fields_values = self._get_command_fields_values(row)
        self._check_all_mandatory_fields_have_values()
        self._process_row(row)

    def _process_row(self, row: Dict[str, Any]) -> NoReturn:
        pass

    def _get_processor_from_field(self, field_name: str) -> Optional[Processor]:
        processor_name = self._fields_values[field_name]
        processor = find_processor_by_name(state=self._glb_idx, processor_name=processor_name)

        if not processor:
            raise CommandExecutionError(f"The processor '{processor_name}' defined in field '{field_name}' "
                                        f"has not been previously declared.")
        return processor

    def _get_interface_from_field(self, field_name: str, processor: Processor) -> Factor:
        interface_name = self._fields_values[field_name]

        if interface_name is None:
            raise CommandExecutionError(f"No interface has been defined for field '{field_name}'.")

        interface = processor.factors_find(interface_name)
        if not interface:
            raise CommandExecutionError(f"The interface '{interface_name}' from has not been found in "
                                        f"processor '{processor.name}'.")

        return interface

    def _get_attributes_from_field(self, field_name: str) -> Dict:
        attributes_list = self._fields_values[field_name]
        attributes = {}
        if attributes_list:
            try:
                attributes = dictionary_from_key_value_list(attributes_list, self._glb_idx)
            except Exception as e:
                raise CommandExecutionError(str(e))

        return attributes


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
