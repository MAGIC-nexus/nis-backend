from typing import Dict, Any, NoReturn

from backend.command_executors import BasicCommand
from backend.command_field_definitions import get_command_fields_from_class
from backend.models.musiasem_concepts import MatrixIndicator


class MatrixIndicatorsCommand(BasicCommand):
    def __init__(self, name: str):
        BasicCommand.__init__(self, name, get_command_fields_from_class(self.__class__))

    def _process_row(self, fields: Dict[str, Any]) -> NoReturn:
        """
        Create and register MatrixIndicator object

        :param fields:
        """
        indicator = MatrixIndicator(
                fields["indicator_name"],
                fields.get("scope"),
                fields.get("processors_selector"),
                fields.get("interfaces_selector"),
                fields.get("indicators_selector"),
                fields.get("attributes_selector"),
                fields.get("description")
        )
        self._glb_idx.put(indicator.key(), indicator)
