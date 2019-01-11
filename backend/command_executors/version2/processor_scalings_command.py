import json
from typing import List, Dict, Union, Any

from backend import IssuesOutputPairType, Issue, CommandField
from backend.command_field_definitions import get_command_fields_from_class
from backend.command_generators.spreadsheet_command_parsers_v2 import IssueLocation
from backend.common.helper import head
from backend.model_services import IExecutableCommand, State, get_case_study_registry_objects


class ProcessorScalingsCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content: Dict = {}
        self._description: str = None

        # internal state properties
        self._current_row_number = 0
        self._issues: List[Issue] = []

        self._command_fields: List[CommandField] = get_command_fields_from_class(self.__class__)

    def execute(self, state: State) -> IssuesOutputPairType:
        # TODO: initialize internal state properties

        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        command_name = self._content["command_name"]

        for row in self._content["items"]:
            self._current_row_number = row["_row"]
            self._process_row(row)

        return self._issues, None

    def _process_row(self, row: Dict[str, Any]):
        fields = self.get_command_fields_values(row)

        if not self.all_mandatory_fields_have_values(fields):
            return

    def get_command_fields_values(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {f.name: row.get(f.name, head(f.allowed_values)) for f in self._command_fields}

    def all_mandatory_fields_have_values(self, fields: Dict[str, Any]) -> bool:
        """Check if mandatory fields without value exist"""
        for field in [f.name for f in self._command_fields if f.mandatory and not fields[f.name]]:
            self.add_issue(3, f"Mandatory field '{field}' is empty. Skipped.")
            return True
        return False

    def add_issue(self, itype: int, description: str):
        self._issues.append(
            Issue(itype=itype,
                  description=description,
                  location=IssueLocation(sheet_name=self._name, row=self._current_row_number, column=None))
        )
        return

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
