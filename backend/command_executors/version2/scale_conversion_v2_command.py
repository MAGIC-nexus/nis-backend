import json
from typing import Optional, Dict, Any

from backend.command_generators import Issue, IssueLocation, IType
from backend.common.helper import strcmp, first
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts import Observer, FactorTypesRelationUnidirectionalLinearTransformObservation, \
    FactorType, Processor
from backend.command_executors import BasicCommand, CommandExecutionError
from backend.command_field_definitions import get_command_fields_from_class
from backend.models.musiasem_concepts_helper import find_or_create_observer, find_processor_by_name


class ScaleConversionV2Command(BasicCommand):
    def __init__(self, name: str):
        BasicCommand.__init__(self, name, get_command_fields_from_class(self.__class__))

    def _process_row(self, fields: Dict[str, Any]) -> None:
        origin_interface_type = self._get_factor_type_from_field("source_hierarchy", "source_interface_type")
        destination_interface_type = self._get_factor_type_from_field("target_hierarchy", "target_interface_type")

        origin_processor: Optional[Processor] = None
        if fields["source_context"]:
            origin_processor = self._get_processor_from_field("source_context")

        destination_processor: Optional[Processor] = None
        if fields["target_context"]:
            destination_processor = self._get_processor_from_field("target_context")

        # Check that the interface types are from different hierarchies (warn if not; not error)
        if origin_interface_type.hierarchy == destination_interface_type.hierarchy:
            self._add_issue(IType.WARNING, f"The interface types '{origin_interface_type.name}' and "
                                           f"'{destination_interface_type.name}' are in the same hierarchy")

        # Create the directed Scale (Linear "Transformation") Relationship
        o = FactorTypesRelationUnidirectionalLinearTransformObservation.create_and_append(
            origin_interface_type, destination_interface_type, fields["scale"],
            origin_processor, destination_processor,
            fields["source_unit"], fields["target_unit"],
            find_or_create_observer(Observer.no_observer_specified, self._glb_idx))

        self._glb_idx.put(o.key(), o)
