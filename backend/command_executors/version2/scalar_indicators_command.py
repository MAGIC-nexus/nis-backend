from typing import Dict, Any, NoReturn

from backend.command_executors import BasicCommand
from backend.command_field_definitions import get_command_fields_from_class
from backend.common.helper import strcmp
from backend.models.musiasem_concepts import Indicator, IndicatorCategories


class ScalarIndicatorsCommand(BasicCommand):
    def __init__(self, name: str):
        BasicCommand.__init__(self, name, get_command_fields_from_class(self.__class__))

    def _process_row(self, fields: Dict[str, Any]) -> NoReturn:
        """
        Create and register Indicator object

        :param fields:
        """
        indicator = Indicator(fields["indicator_name"],
                              fields["formula"],
                              None,
                              fields.get("processors_selector"),
                              None,
                              IndicatorCategories.factors_expression if strcmp(fields.get("local"), "Yes")
                              else IndicatorCategories.case_study,
                              fields.get("description"))
        self._glb_idx.put(indicator.key(), indicator)
