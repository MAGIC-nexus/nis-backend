import json

from backend.command_generators import Issue
from backend.command_generators.basic_elements_parser import dictionary_from_key_value_list
from backend.command_generators.spreadsheet_command_parsers_v2 import IssueLocation
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts import FactorType, Factor, FactorInProcessorType, \
    ProcessorsRelationIsAObservation, ProcessorsRelationPartOfObservation, FactorsRelationDirectedFlowObservation, \
    RelationClassType
from backend.models.musiasem_concepts_helper import create_relation_observations, find_processor_by_name


class RelationshipsCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        def process_line(item):
            r_source_processor_name = item.get("source_processor", None)  # Mandatory, simple_ident
            r_source_interface_name = item.get("source_interface", None)  # Mandatory, simple_ident
            r_target_processor_name = item.get("target_processor", None)  # Mandatory, simple_ident
            r_target_interface_name = item.get("target_interface", None)  # Mandatory, simple_ident
            r_relation_type = item.get("relation_type", None).lower()  # Mandatory, simple_ident
            r_flow_weight = item.get("flow_weight", None)  # Mandatory, simple_ident
            r_source_cardinality = item.get("source_cardinality", None)  # Mandatory, simple_ident
            r_target_cardinality = item.get("target_cardinality", None)  # Mandatory, simple_ident
            r_attributes = item.get("attributes", None)
            if r_attributes:
                try:
                    attributes = dictionary_from_key_value_list(r_attributes)
                except Exception as e:
                    issues.append(Issue(itype=3,
                                        description=str(e),
                                        location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                    return
            else:
                attributes = {}

            r_relation_class = None
            if r_relation_type in ["is_a", "isa"]:
                r_relation_class = RelationClassType.pp_isa
            elif r_relation_type in ["part_of", "partof", "|"]:
                r_relation_class = RelationClassType.pp_part_of
            elif r_relation_type in ["aggregate", "aggregation"]:
                r_relation_class = RelationClassType.pp_aggregate
            elif r_relation_type in ["associate", "association"]:
                r_relation_class = RelationClassType.pp_associate
            elif r_relation_type in ["flow", ">"]:
                r_relation_class = RelationClassType.ff_directed_flow
            elif r_relation_type in ["<"]:
                r_relation_class = RelationClassType.ff_reverse_directed_flow

            if r_relation_type in ["is_a", "isa", "part_of", "partof", "|", "aggregate", "associate"]:
                between_processors = True
            else:  # "flow", ">", "<"
                between_processors = False

            # Look for source Processor
            source_processor = find_processor_by_name(state=glb_idx, processor_name=r_source_processor_name)
            # Look for target Processor
            target_processor = find_processor_by_name(state=glb_idx, processor_name=r_target_processor_name)
            if not between_processors:
                # Look for source InterfaceType
                if r_source_interface_name:
                    source_interface_type = glb_idx.get(FactorType.partial_key(r_source_interface_name))
                    if len(source_interface_type) == 0:
                        source_interface_type = None
                    elif len(source_interface_type) == 1:
                        source_interface_type = source_interface_type[0]
                else:
                    source_interface_type = None
                    
                # Look for target InterfaceType
                if r_target_interface_name:
                    target_interface_type = glb_idx.get(FactorType.partial_key(r_target_interface_name))
                    if len(target_interface_type) == 0:
                        target_interface_type = None
                    elif len(target_interface_type) == 1:
                        target_interface_type = target_interface_type[0]
                else:
                    target_interface_type = None
                if source_interface_type and not target_interface_type:
                    target_interface_type = source_interface_type
                elif not source_interface_type and target_interface_type:
                    source_interface_type = target_interface_type
                elif source_interface_type and target_interface_type:
                    if source_interface_type != target_interface_type:
                        # TODO When different interface types are connected, a scales path should exist (to transform from one type to the other)
                        # TODO Check this and change the type (then, when Scale transform is applied, it will automatically be considered)
                        issues.append(Issue(itype=3,
                                            description="Interface types are not the same (and transformation from one "
                                                        "to the other cannot be performed). Origin: " +
                                                        source_interface_type.name+"; Target: " +
                                                        target_interface_type.name,
                                            location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                        return
                else:  # No interface types!!
                    issues.append(Issue(itype=3,
                                        description="No InterfaceTypes specified or retrieved for a flow",
                                        location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                    return

                # Find source Interface, if not add it
                source_interface = glb_idx.get(Factor.partial_key(processor=source_processor, factor_type=source_interface_type))
                if len(source_interface) == 0:
                    source_interface = None
                elif len(source_interface) == 1:
                    source_interface = source_interface[0]
                if not source_interface:
                    source_interface = Factor.create_and_append(source_interface_type.name,
                                                                source_processor,
                                                                in_processor_type=FactorInProcessorType(external=False,
                                                                                                        incoming=True),
                                                                taxon=source_interface_type)
                    glb_idx.put(source_interface.key(), source_interface)

                # Find target Interface
                target_interface = glb_idx.get(Factor.partial_key(processor=target_processor, factor_type=target_interface_type))
                if len(target_interface) == 0:
                    target_interface = None
                elif len(target_interface) == 1:
                    target_interface = target_interface[0]
                if not target_interface:
                    target_interface = Factor.create_and_append(target_interface_type.name,
                                                                target_processor,
                                                                in_processor_type=FactorInProcessorType(external=False,
                                                                                                        incoming=True),
                                                                taxon=target_interface_type)
                    glb_idx.put(target_interface.key(), target_interface)

            # TODO Pass "attributes" dictionary

            if between_processors:
                create_relation_observations(glb_idx, source_processor, target_processor, r_relation_class, None)
            else:
                create_relation_observations(glb_idx, source_interface, target_interface, r_relation_class, None)

        issues = []
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        name = self._content["command_name"]

        # Process parsed information
        for r, line in enumerate(self._content["items"]):
            process_line(line)

        return issues, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the metadata dictionary
        return self._content

    def json_deserialize(self, json_input):
        # TODO Check validity
        issues = []
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)

        if "description" in json_input:
            self._description = json_input["description"]
        return issues