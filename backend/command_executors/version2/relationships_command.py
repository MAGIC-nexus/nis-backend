import json
import re

from backend.command_executors.execution_helpers import parse_line, classify_variables, \
    obtain_dictionary_with_literal_fields
from backend.command_field_definitions import commands
from backend.command_generators import Issue, parser_field_parsers
from backend.command_generators.parser_field_parsers import string_to_ast, processor_names
from backend.command_generators.parser_ast_evaluators import dictionary_from_key_value_list
from backend.command_generators.spreadsheet_command_parsers_v2 import IssueLocation
from backend.common.helper import create_dictionary, strcmp
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts import FactorType, Factor, FactorInProcessorType, \
    ProcessorsRelationIsAObservation, ProcessorsRelationPartOfObservation, FactorsRelationDirectedFlowObservation, \
    RelationClassType, Parameter, Processor
from backend.models.musiasem_concepts_helper import create_relation_observations, find_processor_by_name
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
            if item["_complex"]:
                asts = parse_line(item, fields)
                if item["_expandable"]:
                    # It is an expandable line
                    # Look for fields which need to be variable in order to originate the expansion
                    res = classify_variables(asts, datasets, hh, parameters)
                    ds_list = res["datasets"]
                    ds_concepts = res["ds_concepts"]
                    h_list = res["hierarchies"]
                    if len(ds_list) >= 1 and len(h_list) >= 1:
                        issues.append(Issue(itype=3,
                                            description="Either a dataset ("+", ".join([d.name for d in ds_list])+") or a hierarchy ("+", ".join([h.name for h in h_list])+")",
                                            location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                        return
                    elif len(ds_list) > 1:
                        issues.append(Issue(itype=3,
                                            description="More than one dataset has been specified: "+", ".join([d.name for d in ds_list]),
                                            location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                        return
                    elif len(h_list) > 1:
                        issues.append(Issue(itype=3,
                                            description="More than one hierarchy has been specified: " + ", ".join(
                                                [h.name for h in h_list]),
                                            location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                        return
                    const_dict = obtain_dictionary_with_literal_fields(item, asts)
                    if len(ds_list) == 1:
                        # If a measure is requested and not all dimensions are used, aggregate or
                        # issue an error because it is not possible to compress.
                        # If only dimensions are used, then obtain all the tuples
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
                        all_dimensions_requested = len(all_dimensions) == 0

                        if measure_requested and not all_dimensions_requested:
                            issues.append(Issue(itype=3,
                                                description="It is not possible to use a measure if not all dimensions are used (cannot assume implicit aggregation)",
                                                location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                            return
                        elif not measure_requested and not all_dimensions_requested:
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

                    # Multiplicity because
                    # - A dataset (only one), iterated. First the concepts are obtained. Then the unique tuples formed by them are obtained.
                    #   - Processor name.
                    # - A set of processors (wildcard or filter by attributes)
                    # - A set of interfaces (according to another filter?)
                    # - Multiple types of relation
                    # - Both (first each dataset record applied -expanded-, then the name evaluation is applied)

                    # - UNRESOLVED: expressions are resolved partially. Parts where parameters
                    # expressions depending on parameters. Only the part of the expression depending on varying things

                    # - The processor name could be a concatenation of multiple literals
                    # -
                    # Look for multiple items in r_source_processor_name, r_source_interface_name,
                    #                            r_target_processor_name, r_target_interface_name
            else:
                print("Single: "+str(item))
                yield item

        def process_line(item):
            r_source_processor_name = item.get("source_processor", None)  # Mandatory, simple_ident
            r_source_interface_name = item.get("source_interface", None)  # Mandatory, simple_ident
            r_target_processor_name = item.get("target_processor", None)  # Mandatory, simple_ident
            r_target_interface_name = item.get("target_interface", None)  # Mandatory, simple_ident
            r_source_name = item.get("source", None)  # Mandatory, simple_ident
            r_target_name = item.get("target", None)  # Mandatory, simple_ident
            r_relation_type = item.get("relation_type", None).lower()  # Mandatory, simple_ident
            r_change_type_scale = item.get("change_type_scale", None)  # Mandatory, simple_ident
            r_flow_weight = item.get("flow_weight", None)  # Mandatory, simple_ident
            r_source_cardinality = item.get("source_cardinality", None)  # Mandatory, simple_ident
            r_target_cardinality = item.get("target_cardinality", None)  # Mandatory, simple_ident
            r_attributes = item.get("attributes", None)
            if r_attributes:
                try:
                    attributes = dictionary_from_key_value_list(r_attributes, glb_idx)
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
                # Look for source Interface
                # First find an Interface in the Processor by that name
                source_interface = None
                if r_source_interface_name:
                    for f in source_processor.factors:
                        if strcmp(f.name, r_source_interface_name):
                            source_interface = f
                target_interface = None
                if r_target_interface_name:
                    for f in target_processor.factors:
                        if strcmp(f.name, r_target_interface_name):
                            target_interface = f

                # If not, look for an InterfaceType
                if not source_interface:
                    if r_source_interface_name:
                        source_interface_type = glb_idx.get(FactorType.partial_key(r_source_interface_name))
                        if len(source_interface_type) == 0:
                            source_interface_type = None
                        elif len(source_interface_type) == 1:
                            source_interface_type = source_interface_type[0]
                    else:
                        source_interface_type = None
                else:
                    source_interface_type = source_interface.taxon
                    
                # Look for target InterfaceType
                if not target_interface:
                    if r_target_interface_name:
                        target_interface_type = glb_idx.get(FactorType.partial_key(r_target_interface_name))
                        if len(target_interface_type) == 0:
                            target_interface_type = None
                        elif len(target_interface_type) == 1:
                            target_interface_type = target_interface_type[0]
                    else:
                        target_interface_type = None
                else:
                    target_interface_type = target_interface.taxon

                change_type_scale = None
                if source_interface_type and not target_interface_type:
                    target_interface_type = source_interface_type
                elif not source_interface_type and target_interface_type:
                    source_interface_type = target_interface_type
                elif source_interface_type and target_interface_type:
                    if source_interface_type != target_interface_type:
                        # TODO When different interface types are connected, a scales path should exist (to transform from one type to the other)
                        # TODO Check this and change the type (then, when Scale transform is applied, it will automatically be considered)
                        if not r_change_type_scale:
                            issues.append(Issue(itype=3,
                                                description="Interface types are not the same (and transformation from one "
                                                            "to the other cannot be performed). Origin: " +
                                                            source_interface_type.name+"; Target: " +
                                                            target_interface_type.name,
                                                location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                            return
                        else:
                            change_type_scale = r_change_type_scale
                else:  # No interface types!!
                    issues.append(Issue(itype=3,
                                        description="No InterfaceTypes specified or retrieved for a flow",
                                        location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                    return

                # Find source Interface, if not add it
                if not source_interface:
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
                if not target_interface:
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

            # TODO Pass full "attributes" dictionary
            # Pass "change_type_scale" as attribute
            if change_type_scale:
                attributes = dict(_change_type_scale=change_type_scale)
            else:
                attributes = None

            if between_processors:
                create_relation_observations(glb_idx, source_processor, target_processor, r_relation_class, None, attributes=attributes)
            else:
                create_relation_observations(glb_idx, source_interface, target_interface, r_relation_class, None, attributes=attributes)

        fields = {f.name: f for f in commands["Relationships"]}

        issues = []
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        name = self._content["command_name"]

        # Obtain the names of all parameters
        parameters = [p.name for p in glb_idx.get(Parameter.partial_key())]
        # Obtain the names of all processors
        all_processors = get_processor_names_to_processors_dictionary(glb_idx)

        # Process parsed information
        for r, line in enumerate(self._content["items"]):
            print(str(r))
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