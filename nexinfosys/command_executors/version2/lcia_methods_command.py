import json
from typing import Optional, Dict, Any

from nexinfosys.command_generators import Issue, IssueLocation, IType
from nexinfosys.common.helper import strcmp, first, PartialRetrievalDictionary
from nexinfosys.model_services import IExecutableCommand, get_case_study_registry_objects
from nexinfosys.models.musiasem_concepts import Observer, FactorTypesRelationUnidirectionalLinearTransformObservation, \
    FactorType, Processor, Indicator
from nexinfosys.command_executors import BasicCommand, CommandExecutionError, subrow_issue_message
from nexinfosys.command_field_definitions import get_command_fields_from_class
from nexinfosys.models.musiasem_concepts_helper import find_or_create_observer, find_processor_by_name


class LCIAMethodsCommand(BasicCommand):
    def __init__(self, name: str):
        BasicCommand.__init__(self, name, get_command_fields_from_class(self.__class__))

    def _process_row(self, fields: Dict[str, Any], subrow=None) -> None:
        """
        :param fields:
        :param subrow:
        :return:
        """
        # InterfaceType must exist
        try:
            self._get_factor_type_from_field(None, "interface")
        except CommandExecutionError as e:
            self._add_issue(IType.ERROR, str(e))

        # (LCIA) Indicator must exist
        indicator = self._glb_idx.get(Indicator.partial_key(fields["lcia_indicator"]))
        if len(indicator) == 1:
            pass
        elif len(indicator) == 0:
            self._add_issue(IType.ERROR, f"Indicator with name '{fields['lcia_indicator']}' not found" + subrow_issue_message(subrow))
            return
        else:
            self._add_issue(IType.WARNING,
                            f"Indicator with name '{fields['lcia_indicator']}' found {len(indicator)} times" + subrow_issue_message(subrow))
            return

        # Store LCIA Methods as a new variable.
        # TODO Use it to prepare a pd.DataFrame previous to calculating Indicators (after solving). Use "to_pickable"
        lcia_methods = self._state.get("_lcia_methods")
        if not lcia_methods:
            lcia_methods = PartialRetrievalDictionary()
            self._state.set("_lcia_methods", lcia_methods)
        _ = dict(m=fields["lcia_method"],
                 d=fields["lcia_indicator"],
                 h=fields["lcia_horizon"],
                 i=fields["interface"])
        lcia_methods.put(_, (fields["interface_unit"], fields["lcia_coefficient"]))

