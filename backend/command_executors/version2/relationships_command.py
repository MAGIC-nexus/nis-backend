import json
import re
from typing import Tuple, Optional, NoReturn, List

from backend.command_executors.execution_helpers import parse_line, classify_variables, \
    obtain_dictionary_with_literal_fields
from backend.command_field_definitions import get_command_fields_from_class
from backend.command_generators import Issue, IssueLocation, IType
from backend.command_generators.parser_ast_evaluators import dictionary_from_key_value_list
from backend.command_generators.parser_field_parsers import string_to_ast, processor_names
from backend.common.helper import strcmp
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts import FactorType, Factor, FactorInProcessorType, \
    RelationClassType, Parameter, Processor, FactorTypesRelationUnidirectionalLinearTransformObservation, \
    FactorTypesConverter
from backend.models.musiasem_concepts_helper import create_relation_observations, find_processor_by_name, \
    find_or_create_factor, find_factor_type_scale_relation
from backend.solving import get_processor_names_to_processors_dictionary


def obtain_matching_processors(parsed_processor_name, all_processors):
    """

    :param parsed_processor_name: The AST of parsing processor names (rule "processor_names")
    :param all_processors: either a list with all processor names or a dict with full processor name to Processor
    :return: either the set of processor names matching the filter or the set of Processor whose names match the filter
    """
    # Prepare "processor_name"
    s = r""
    first = True
    for p in parsed_processor_name["parts"]:
        if p[0] == "separator":
            if p[1] == "..":
                if first:
                    s += r".*"
                    if len(parsed_processor_name["parts"]) > 1:
                        s += r"\."
                else:
                    s += r"\..*"
            else:
                s += r"\."
        else:
            s += p[1]
        first = False
    reg = re.compile(s)
    res = set()
    add_processor = isinstance(all_processors, dict)
    for p in all_processors:
        if reg.match(p):
            if add_processor:
                res.add(all_processors[p])
            else:
                res.add(p)

    return res


class RelationshipsCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        def parse_and_unfold_line(item):
            # Consider multiplicity because of:
            # - A dataset (only one). First a list of dataset concepts used in the line is obtained.
            #   Then the unique tuples formed by them are obtained.
            # - Processor name.
            #   - A set of processors (wildcard or filter by attributes)
            #   - A set of interfaces (according to another filter?)
            # - Multiple types of relation
            # - Both (first each dataset record applied -expanded-, then the name evaluation is applied)
            # - UNRESOLVED: expressions are resolved partially. Parts where parameters
            # expressions depending on parameters. Only the part of the expression depending on varying things
            # - The processor name could be a concatenation of multiple literals
            #
            # Look for multiple items in r_source_processor_name, r_source_interface_name,
            #                            r_target_processor_name, r_target_interface_name
            if item["_complex"]:
                asts = parse_line(item, fields)
                if item["_expandable"]:
                    # It is an expandable line
                    # Look for fields which are specified to be variable in order to originate the expansion
                    res = classify_variables(asts, datasets, hh, parameters)
                    ds_list = res["datasets"]
                    ds_concepts = res["ds_concepts"]
                    h_list = res["hierarchies"]
                    if len(ds_list) >= 1 and len(h_list) >= 1:
                        issues.append(Issue(itype=IType.ERROR,
                                            description="Dataset(s): "+", ".join([d.name for d in ds_list])+", and hierarchy(ies): "+", ".join([h.name for h in h_list])+", have been specified. Either a single dataset or a single hiearchy is supported.",
                                            location=IssueLocation(sheet_name=name, row=r, column=None)))
                        return
                    elif len(ds_list) > 1:
                        issues.append(Issue(itype=IType.ERROR,
                                            description="More than one dataset has been specified: "+", ".join([d.name for d in ds_list])+", just one dataset is supported.",
                                            location=IssueLocation(sheet_name=name, row=r, column=None)))
                        return
                    elif len(h_list) > 1:
                        issues.append(Issue(itype=IType.ERROR,
                                            description="More than one hierarchy has been specified: " + ", ".join([h.name for h in h_list])+", just one hierarchy is supported.",
                                            location=IssueLocation(sheet_name=name, row=r, column=None)))
                        return
                    const_dict = obtain_dictionary_with_literal_fields(item, asts)
                    if len(ds_list) == 1:
                        # If a measure is requested and not all dimensions are used, aggregate or
                        # issue an error (because it is not possible to reduce without aggregation).
                        # If only dimensions are used, then obtain all the unique tuples
                        ds = ds_list[0]
                        measure_requested = False
                        all_dimensions = set([c.code for c in ds.dimensions if not c.is_measure])
                        for con in ds_concepts:
                            for c in ds.dimensions:
                                if strcmp(c.code, con):
                                    if c.is_measure:
                                        measure_requested = True
                                    else:  # Dimension
                                        all_dimensions.remove(c.code)
                        only_dimensions_requested = len(all_dimensions) == 0

                        if measure_requested and not only_dimensions_requested:
                            issues.append(Issue(itype=IType.ERROR,
                                                description="It is not possible to use a measure if not all dimensions are used (cannot assume implicit aggregation)",
                                                location=IssueLocation(sheet_name=name, row=r, column=None)))
                            return
                        elif not measure_requested and not only_dimensions_requested:
                            # TODO Reduce the dataset to the unique tuples (consider the current case -sensitive or not-sensitive-)
                            data = None
                        else:  # Take the dataset as-is
                            data = ds.data

                        for row in data.iterrows():
                            item2 = const_dict.copy()

                            d = {}
                            for c in ds_concepts:
                                d["{" + ds.code + "." + c + "}"] = row[c]
                            # Expand in all fields
                            for f in fields:
                                if f not in const_dict:
                                    # Replace all
                                    string = item[f]
                                    # TODO Could iterate through the variables in the field (not IN ALL FIELDS of the row)
                                    for item in sorted(d.keys(), key=len, reverse=True):
                                        string = re.sub(item, d[item], string)
                                    item2[f] = string
                            # Now, look for wildcards where it is allowed
                            r_source_processor_name = string_to_ast(processor_names, item2.get("source_processor", None))
                            r_target_processor_name = string_to_ast(processor_names, item2.get("target_processor", None))
                            if ".." in r_source_processor_name or ".." in r_target_processor_name:
                                if ".." in r_source_processor_name:
                                    source_processor_names = obtain_matching_processors(r_source_processor_name, all_processors)
                                else:
                                    source_processor_names = [r_source_processor_name]
                                if ".." in r_target_processor_name:
                                    target_processor_names = obtain_matching_processors(r_target_processor_name, all_processors)
                                else:
                                    target_processor_names = [r_target_processor_name]
                                for s in source_processor_names:
                                    for t in target_processor_names:
                                        item3 = item2.copy()
                                        item3["source_processor"] = s
                                        item3["target_processor"] = t
                                        print("Multiple by dataset and wildcard: " + str(item3))
                                        yield item3
                            else:
                                print("Multiple by dataset: " + str(item3))
                                yield item2
                    elif len(h_list) == 1:
                        pass
                    else:  # No dataset, no hierarchy of categories, but still complex, because of wildcards
                        wildcard_in_source = ".." in item.get("source_processor", "")
                        wildcard_in_target = ".." in item.get("target_processor", "")
                        if wildcard_in_source or wildcard_in_target:
                            r_source_processor_name = string_to_ast(processor_names, item.get("source_processor", None))
                            r_target_processor_name = string_to_ast(processor_names, item.get("target_processor", None))
                            if wildcard_in_source:
                                source_processor_names = obtain_matching_processors(r_source_processor_name, all_processors)
                            else:
                                source_processor_names = [item["source_processor"]]
                            if wildcard_in_target:
                                target_processor_names = obtain_matching_processors(r_target_processor_name, all_processors)
                            else:
                                target_processor_names = [item["target_processor"]]
                            for s in source_processor_names:
                                for t in target_processor_names:
                                    item3 = const_dict.copy()
                                    item3["source_processor"] = s
                                    item3["target_processor"] = t
                                    print("Multiple by wildcard: "+str(item3))
                                    yield item3
                        else:
                            # yield item
                            raise Exception("If 'complex' is signaled, it should not pass by this line")
            else:
                # print("Single: "+str(item))
                yield item

        def check_flow_orientation(source_processor: Processor, target_processor: Processor,
                                   source_interface: Factor, target_interface: Factor, is_direct_flow: bool):
            allowed_source_orientation = ("Output" if is_direct_flow else "Input")
            error_issues: List[str] = []

            # Are the orientations equal?
            if strcmp(source_interface.orientation, target_interface.orientation):
                if strcmp(source_interface.orientation, allowed_source_orientation):
                    # Target processor should be parent of source processor
                    parent_processor = target_processor
                    child_processor = source_processor
                else:
                    # Source processor should be parent of target processor
                    parent_processor = source_processor
                    child_processor = target_processor

                if child_processor not in parent_processor.children(glb_idx):
                    error_issues.append(f"The processor '{child_processor.name}' should be part of the "
                                        f"processor '{parent_processor.name}' when using the same interface "
                                        f"orientation '{source_interface.orientation}'.")

            else:  # Orientations are different
                if not strcmp(source_interface.orientation, allowed_source_orientation):
                    error_issues.append(f"The source interface '{source_interface.full_name}' has the wrong "
                                        f"orientation '{source_interface.orientation}'.")

                if strcmp(target_interface.orientation, allowed_source_orientation):
                    error_issues.append(f"The target interface '{target_interface.full_name}' has the wrong "
                                        f"orientation '{target_interface.orientation}'.")

            return error_issues

        def process_line(item):
            r_source_processor_name = item.get("source_processor")  # Mandatory, simple_ident
            r_source_interface_name = item.get("source_interface")  # Mandatory, simple_ident
            r_target_processor_name = item.get("target_processor")  # Mandatory, simple_ident
            r_target_interface_name = item.get("target_interface")  # Mandatory, simple_ident
            r_relation_type = item.get("relation_type").lower()  # Mandatory, simple_ident
            r_change_type_scale = item.get("change_type_scale")  # Mandatory, simple_ident
            r_flow_weight = item.get("flow_weight")  # Mandatory, simple_ident
            r_source_cardinality = item.get("source_cardinality")  # Mandatory, simple_ident
            r_target_cardinality = item.get("target_cardinality")  # Mandatory, simple_ident
            r_attributes = item.get("attributes")
            if r_attributes:
                try:
                    attributes = dictionary_from_key_value_list(r_attributes, glb_idx)
                except Exception as e:
                    add_issue(IType.ERROR, str(e))
                    return
            else:
                attributes = {}

            r_relation_class = RelationClassType.from_str(r_relation_type)

            if r_relation_class in [RelationClassType.ff_directed_flow,
                                    RelationClassType.ff_reverse_directed_flow,
                                    RelationClassType.ff_scale]:
                between_processors = False
            else:
                between_processors = True

            # Look for source and target Processors
            source_processor = find_processor_by_name(state=glb_idx, processor_name=r_source_processor_name)
            if not source_processor:
                add_issue(IType.ERROR, f"The source processor '{r_source_processor_name}' doesn't exist.")
                return

            target_processor = find_processor_by_name(state=glb_idx, processor_name=r_target_processor_name)
            if not target_processor:
                add_issue(IType.ERROR, f"The target processor '{r_target_processor_name}' doesn't exist.")
                return

            change_type_scale: Optional[FactorTypesConverter] = None

            if not between_processors:
                # Look for source and target Interfaces
                source_interface = source_processor.factors_find(r_source_interface_name)
                if not source_interface:
                    add_issue(IType.ERROR, f"The source interface '{r_source_interface_name}' has not been found in "
                                           f"processor '{source_processor.name}'.")
                    return

                target_interface = target_processor.factors_find(r_target_interface_name)
                if not target_interface:
                    add_issue(IType.ERROR, f"The target interface '{r_target_interface_name}' has not been found in "
                                           f"processor '{target_processor.name}'.")
                    return

                # Check for correct interfaces orientation (input/output) of source and target
                if r_relation_class in [RelationClassType.ff_directed_flow, RelationClassType.ff_reverse_directed_flow]:
                    orientation_issues = check_flow_orientation(
                        source_processor, target_processor, source_interface, target_interface,
                        is_direct_flow=(r_relation_class == RelationClassType.ff_directed_flow)
                    )

                    if len(orientation_issues) > 0:
                        for description in orientation_issues:
                            add_issue(IType.ERROR, description)

                        return

                if source_interface.taxon != target_interface.taxon:
                    if not r_change_type_scale:
                        # Check if a scale change has been specified
                        scale_relations = find_factor_type_scale_relation(
                            glb_idx, source_interface.taxon, target_interface.taxon, source_processor, target_processor)

                        if len(scale_relations) == 0:
                            add_issue(IType.ERROR, f"Interface types are not the same (and transformation from one "
                                                   f"to the other cannot be performed). Origin: "
                                                   f"{source_interface.taxon.name}; Target: {target_interface.taxon.name}")
                            return
                        elif len(scale_relations) > 1:
                            add_issue(IType.ERROR,
                                      f"Multiple transformations can be applied between interfaces. Origin: "
                                      f"{source_interface.taxon.name}; Target: {target_interface.taxon.name}")
                            return
                        else:
                            change_type_scale = scale_relations[0].converter
                    else:
                        change_type_scale = FactorTypesConverter(r_change_type_scale)

            # TODO Pass full "attributes" dictionary
            # Pass "change_type_scale" as attribute
            if change_type_scale:
                attributes = dict(_change_type_scale=change_type_scale)
            else:
                attributes = None

            if between_processors:
                create_relation_observations(glb_idx,
                                             source_processor,
                                             [(p, r_relation_class) for p in target_processor] if isinstance(
                                                 target_processor, list) else [(target_processor, r_relation_class)],
                                             r_relation_class,
                                             None,
                                             attributes=attributes
                                             )
            else:
                create_relation_observations(glb_idx,
                                             source_interface,
                                             [(i, r_relation_class, r_flow_weight) for i in target_interface] if isinstance(
                                                 target_interface, list) else [(target_interface, r_relation_class, r_flow_weight)],
                                             r_relation_class,
                                             None,
                                             attributes=attributes
                                             )

        def add_issue(itype: IType, description: str) -> NoReturn:
            issues.append(Issue(itype=itype,
                                description=description,
                                location=IssueLocation(sheet_name=name, row=r, column=None)))

        fields = {f.name: f for f in get_command_fields_from_class(self.__class__)}

        issues: List[Issue] = []
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        name = self._content["command_name"]

        # Obtain the names of all parameters
        parameters = [p.name for p in glb_idx.get(Parameter.partial_key())]
        # Obtain the names of all processors
        all_processors = get_processor_names_to_processors_dictionary(glb_idx)

        # Process parsed information
        for line in self._content["items"]:
            r = line["_row"]
            for sub_line in parse_and_unfold_line(line):
                process_line(sub_line)

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
