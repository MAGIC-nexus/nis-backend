import json
from typing import List, Dict, Union, Any, Optional, Tuple

from backend import IssuesOutputPairType, CommandField
from backend.command_field_definitions import get_command_fields_from_class
from backend.common.helper import head, strcmp, PartialRetrievalDictionary
from backend.model_services import IExecutableCommand, State, get_case_study_registry_objects
from backend.command_generators import Issue, IType, IssueLocation
from backend.command_generators.parser_ast_evaluators import ast_evaluator
from backend.command_generators.parser_field_parsers import string_to_ast, expression_with_parameters
from backend.models.musiasem_concepts import Processor, ProcessorsRelationPartOfObservation, Factor, \
    ProcessorsRelationUpscaleObservation
from backend.models.musiasem_concepts_helper import find_processor_by_name


class CommandExecutionError(Exception):
    pass


class ProcessorScalingsCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content: Dict = {}
        self._command_name = ""
        self._command_fields: List[CommandField] = get_command_fields_from_class(self.__class__)

        # Execution state
        self._issues: List[Issue] = None
        self._current_row_number: int = None
        self._glb_idx: PartialRetrievalDictionary = None
        self._fields_values = {}

    def _get_command_fields_values(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {f.name: row.get(f.name, head(f.allowed_values)) for f in self._command_fields}

    def _check_all_mandatory_fields_have_values(self):
        empty_fields: List[str] = [f.name for f in self._command_fields if f.mandatory and not self._fields_values[f.name]]

        if len(empty_fields) > 0:
            raise CommandExecutionError(f"Mandatory field/s '{', '.join(empty_fields)}' is/are empty.")

    def _add_issue(self, itype: int, description: str):
        self._issues.append(
            Issue(itype=itype,
                  description=description,
                  location=IssueLocation(sheet_name=self._command_name,
                                         row=self._current_row_number, column=None))
        )
        return

    def estimate_execution_time(self):
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

    def execute(self, state: State) -> IssuesOutputPairType:
        self._issues = []

        self._glb_idx, _, _, _, _ = get_case_study_registry_objects(state)

        for row in self._content["items"]:
            try:
                self._process_row(row)
            except CommandExecutionError as e:
                self._add_issue(IType.error(), str(e))

        return self._issues, None

    def _process_row(self, row: Dict[str, Any]):
        self._current_row_number = row["_row"]
        self._fields_values = self._get_command_fields_values(row)

        self._check_all_mandatory_fields_have_values()

        scaling_type = self._fields_values["scaling_type"]
        scale: str = self._fields_values["scale"]

        # Find processors
        invoking_processor = self._get_processor_from_field("invoking_processor")
        requested_processor = self._get_processor_from_field("requested_processor")

        invoking_interface_name: str = self._fields_values["invoking_interface"]
        requested_interface_name: str = self._fields_values["requested_interface"]

        if strcmp(scaling_type, "CloneAndScale"):
            # TODO: check “RequestedProcessor” must be an archetype
            # 1. Clones “RequestedProcessor” as a child of “InvokingProcessor”
            requested_processor_clone = self._clone_processor_as_child(processor=requested_processor,
                                                                       parent_processor=invoking_processor)

            # 2. Constrains the value of “RequestedInterface” to the value of “InvokingInterface”, scaled by “Scale”
            # TODO: change the constraining to use the "requested_interface_name"
            self._constrains_interface(scale=scale, interface_name=invoking_interface_name,
                                       parent_processor=invoking_processor,
                                       child_processor=requested_processor_clone)

        elif strcmp(scaling_type, "Scale"):
            # Processors must be of same type (archetype or instance)
            if not strcmp(invoking_processor.instance_or_archetype, requested_processor.instance_or_archetype):
                raise CommandExecutionError("Requested and invoking processors should be of the same type "
                                            "(both instance or_archetype)")

            # 1. Constrains the value of “RequestedInterface” to the value of “InvokingInterface”, scaled by “Scale”
            # TODO: change the constraining to use the "requested_interface_name"
            self._constrains_interface(scale=scale, interface_name=invoking_interface_name,
                                       parent_processor=invoking_processor,
                                       child_processor=requested_processor)

        elif strcmp(scaling_type, "CloneScaled"):
            # “RequestedProcessor” must be an archetype
            # if not strcmp(requested_processor.instance_or_archetype, "archetype"):
            #     raise CommandExecutionError(f"Requested processor '{requested_processor.name}' should be of type 'archetype'")

            # “InvokingProcessor” must be an instance
            # if not strcmp(invoking_processor.instance_or_archetype, "instance"):
            #     raise CommandExecutionError(f"Invoking processor '{invoking_processor.name}' should be of type 'instance'")

            # 1. Clones “RequestedProcessor” as a child of “InvokingProcessor”
            # 2. Scales the new processor using “Scale” as the value of “RequestedInterface”
            requested_processor_clone = self._clone_processor_as_child(processor=requested_processor,
                                                                       parent_processor=invoking_processor)

            # Value Scale, which can be an expression, should be evaluated (ast) because we need a final float number
            scale_value = self._get_scale_value(scale)

            # In the cloned processor search in all interfaces if there are Observations relative_to RequestedInterface
            # and multiply the observation by the computed scale.
            self._scale_observations_relative_to_interface(processor=requested_processor_clone,
                                                           interface_name=requested_interface_name,
                                                           scale=scale_value)

    def _get_processor_from_field(self, field_name: str) -> Optional[Processor]:
        processor_name = self._fields_values[field_name]
        processor = find_processor_by_name(state=self._glb_idx, processor_name=processor_name)

        if not processor:
            raise CommandExecutionError(f"The processor '{processor_name}' defined in field '{field_name}' "
                                        f"has not been previously declared.")
        return processor

    def _clone_processor_as_child(self, processor: Processor, parent_processor: Processor) -> Processor:
            processor_clone = processor.clone(state=self._glb_idx)

            # Create PART-OF relation
            relationship = ProcessorsRelationPartOfObservation.create_and_append(parent=parent_processor,
                                                                                 child=processor_clone)
            self._glb_idx.put(relationship.key(), relationship)

            # Add cloned processor hierarchical names to global index
            for hierarchical_name in processor_clone.full_hierarchy_names(self._glb_idx):
                self._glb_idx.put(Processor.partial_key(name=hierarchical_name, ident=processor_clone.ident),
                                  processor_clone)

            return processor_clone

    def _constrains_interface(self, scale: str, interface_name: str, parent_processor: Processor, child_processor: Processor):
        relationship = ProcessorsRelationUpscaleObservation.create_and_append(parent=parent_processor,
                                                                              child=child_processor,
                                                                              observer=None,
                                                                              factor_name=interface_name,
                                                                              quantity=scale)

        self._glb_idx.put(relationship.key(), relationship)

    def _get_scale_value(self, scale: str):
        try:
            value = float(scale)
        except ValueError:
            ast = string_to_ast(expression_with_parameters, scale)

            evaluation_issues: List[Tuple[int, str]] = []
            value, unresolved_vars = ast_evaluator(exp=ast, state=self._glb_idx, obj=None, issue_lst=evaluation_issues)

            if len(evaluation_issues) > 0:
                evaluation_issues_str = [i[1] for i in evaluation_issues]
                raise CommandExecutionError(f"Problems evaluating scale expression '{scale}': "
                                            f"{', '.join(evaluation_issues_str)}")
            elif len(unresolved_vars) > 0:
                raise CommandExecutionError(f"Unresolved variables evaluating the scale expression '{scale}':"
                                            f" {', '.join(unresolved_vars)}")

            elif not value:
                raise CommandExecutionError(f"The scale expression '{scale}' could not be evaluated.")

        return value

    def _scale_observations_relative_to_interface(self, processor: Processor, interface_name: str,
                                                  scale: Union[int, float]):
        for factor in processor.factors:
            for observation in factor.quantitative_observations:
                relative_to_interface = observation.attributes.get("relative_to", None)
                if relative_to_interface and strcmp(relative_to_interface.name, interface_name):
                    observation.value = float(observation.value) * scale
                    observation.attributes["relative_to"] = None

