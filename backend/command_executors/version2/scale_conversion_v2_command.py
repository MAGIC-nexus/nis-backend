import json
from typing import NoReturn, Optional

from backend.command_generators import Issue, IssueLocation, IType
from backend.common.helper import strcmp
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts import Observer, FactorTypesRelationUnidirectionalLinearTransformObservation, \
    FactorType, Processor
from models.musiasem_concepts_helper import find_or_create_observer, find_processor_by_name


class ScaleConversionV2Command(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        def add_issue(itype: IType, description: str) -> NoReturn:
            issues.append(Issue(itype=itype,
                                description=description,
                                location=IssueLocation(sheet_name=name, row=r, column=None)))

        def get_factor_type(hierarchy_name: str, interface_type_name: str, orig_or_dest: str) -> Optional[FactorType]:
            if not interface_type_name:
                add_issue(IType.ERROR, f"The {orig_or_dest} interface type name has not been specified")
                return None

            # Check if FactorType exists
            ft_list = glb_idx.get(FactorType.partial_key(interface_type_name))
            if len(ft_list) > 0:
                if len(ft_list) == 1:
                    return ft_list[0]
                else:
                    if not hierarchy_name:
                        add_issue(IType.ERROR,
                                  f"The hierarchy of the {orig_or_dest} interface type {interface_type_name} "
                                  f"has not been specified and this name is not unique")
                        return None

                    for ft in ft_list:
                        if strcmp(ft.hierarchy.name, hierarchy_name):
                            return ft

        def get_processor(processor_name: str) -> Optional[Processor]:
            processor = find_processor_by_name(state=glb_idx, processor_name=processor_name)

            if not processor:
                add_issue(IType.ERROR, f"The processor '{processor_name}' has not been previously declared.")

            return processor

        def process_line(item):
            sc_src_hierarchy = item.get("source_hierarchy")
            sc_src_interface_type = item.get("source_interface_type")
            sc_tgt_hierarchy = item.get("target_hierarchy")
            sc_tgt_interface_type = item.get("target_interface_type")
            sc_scale = item.get("scale")
            sc_src_context = item.get("source_context")
            sc_tgt_context = item.get("target_context")
            sc_src_unit = item.get("source_unit")
            sc_tgt_unit = item.get("target_unit")

            origin_interface_type = get_factor_type(sc_src_hierarchy, sc_src_interface_type, "origin")
            destination_interface_type = get_factor_type(sc_tgt_hierarchy, sc_tgt_interface_type, "destination")

            if not origin_interface_type or not destination_interface_type:
                return

            origin_processor: Optional[Processor] = None
            if sc_src_context:
                origin_processor = get_processor(sc_src_context)
                if not origin_processor:
                    return

            destination_processor: Optional[Processor] = None
            if sc_tgt_context:
                destination_processor = get_processor(sc_tgt_context)
                if not destination_processor:
                    return

            # Check that the interface types are from different hierarchies (warn if not; not error)
            if origin_interface_type.hierarchy == destination_interface_type.hierarchy:
                add_issue(IType.WARNING, f"The interface types '{origin_interface_type.name}' and "
                                         f"'{destination_interface_type.name}' are in the same hierarchy")

            # Create the directed Scale (Linear "Transformation") Relationship
            o = FactorTypesRelationUnidirectionalLinearTransformObservation.create_and_append(
                origin_interface_type, destination_interface_type, sc_scale, origin_processor, destination_processor,
                sc_src_unit, sc_tgt_unit, find_or_create_observer(Observer.no_observer_specified, glb_idx))

            glb_idx.put(o.key(), o)

        issues = []
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        name = self._content["command_name"]

        # Process parsed information
        for line in self._content["items"]:
            r = line["_row"]
            # If the line contains a reference to a dataset or hierarchy, expand it
            # If not, process it directly
            is_expansion = False
            if is_expansion:
                # TODO Iterate through dataset and/or hierarchy elements, producing a list of new items
                pass
            else:
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