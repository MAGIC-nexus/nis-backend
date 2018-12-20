import json
import re

from backend import CommandField
from backend.command_executors.execution_helpers import parse_line, classify_variables, \
    obtain_dictionary_with_literal_fields
from backend.command_executors.version2.relationships_command import obtain_matching_processors
from backend.command_field_definitions import processor_types, functional_or_structural, instance_or_archetype, \
    copy_interfaces_mode, commands
from backend.command_generators import Issue
from backend.command_generators.parser_ast_evaluators import dictionary_from_key_value_list
from backend.command_generators.parser_field_parsers import string_to_ast, processor_names
from backend.command_generators.spreadsheet_command_parsers_v2 import IssueLocation
from backend.common.helper import first
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts import ProcessorsSet, ProcessorsRelationPartOfObservation, Parameter, Processor, \
    Geolocation
from backend.models.musiasem_concepts_helper import find_or_create_processor, find_processor_by_name
from backend.solving import get_processor_names_to_processors_dictionary


def get_object_view(d):
    class objectview(object):
        def __init__(self, d2):
            self.__dict__ = d2
    return objectview(d)


class ProcessorsCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        """
        Create empty Processors, potentially related with PartOf relationship

        The Name of a new Processor can be:
        * Simple name
        * Hierarchical name
        * Clone previously declared processor (and children), where "previously declared processor" can be simple or complex

        If name is hierarchic, a formerly declared processor is assumed.
        If a parent processor is specified, the processor will be related to the new parent (if parents are different -> multiple functionality)
        In a hierarchy, children inherit interfaces from parents
                        if parents do not have interfaces, do they "receive" interfaces from immediate child?
                        - Only if children have them already (separate copy)
                        - If children do not have them already (connected copy)

        If CLONE(<existing processor>) is specified, a copy of the <existing processor> (and children) will be created.
        In this case, the Parent field is mandatory (if not, no unique name will be available). For a pair "parent processor - clone processor", this operation is unique (unless, a naming syntax for the copy is invented)
        If child processors had interface, copy them into the parent processor (the default move is downwards, from parent to child)

        Later, when interfaces are attached to processors,
        :param state:
        :return:
        """
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
                        issues.append(create_issue(3, "Dataset(s): "+", ".join([d.name for d in ds_list])+", and hierarchy(ies): "+", ".join([h.name for h in h_list])+", have been specified. Either a single dataset or a single hiearchy is supported."))
                        return
                    elif len(ds_list) > 1:
                        issues.append(create_issue(3, "More than one dataset has been specified: "+", ".join([d.name for d in ds_list])+", just one dataset is supported."))
                        return
                    elif len(h_list) > 1:
                        issues.append(create_issue(3, "More than one hierarchy has been specified: " + ", ".join([h.name for h in h_list])+", just one hierarchy is supported."))
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
                            issues.append(create_issue(3, "It is not possible to use a measure if not all dimensions are used (cannot assume implicit aggregation)"))
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
                print("Single: "+str(item))
                yield item

        def process_line(item):
            fields_dict = {f.name: item.get(f.name, first(f.allowed_values)) for f in commands["Processors"]}

            # Check if mandatory fields with no value exist
            for field in [f.name for f in commands["Processors"] if f.mandatory and not fields_dict[f.name]]:
                issues.append(create_issue(3, f"Mandatory field '{field}' is empty. Skipped."))
                return

            # # Processor must have a name
            # if not fields_dict["name"]:
            #     issues.append(Issue(itype=3,
            #                         description="Empty processor name. Skipped.",
            #                         location=IssueLocation(sheet_name=command_name, row=row, column=None)))
            #     return

            # Transform text of "attributes" into a dictionary
            field_value = fields_dict.get("attributes", None)
            if field_value:
                try:
                    fields_dict["attributes"] = dictionary_from_key_value_list(field_value, glb_idx)
                except Exception as e:
                    issues.append(create_issue(3, str(e)))
                    return
            else:
                fields_dict["attributes"] = {}

            #
            # # Read variables
            # p_name = item.get("processor", None)  # Mandatory, simple_ident
            # p_group = item.get("processor_group", None)  # Optional, simple_ident
            # p_type = item.get("processor_type", processor_types[0])  # Optional, simple_ident
            # p_f_or_s = item.get("functional_or_structural", functional_or_structural[0])  # Optional, simple_ident
            # p_copy_interfaces = item.get("copy_interfaces_mode", copy_interfaces_mode[0])
            # p_i_or_a = item.get("instance_or_archetype", instance_or_archetype[0])  # Optional, simple_ident
            # p_parent = item.get("parent_processor", None)  # Optional, simple_ident
            # p_clone_processor = item.get("clone_processor", None)  # Optional, simple_ident
            # p_alias = item.get("alias", None)  # Optional, simple_ident
            # p_description = item.get("description", None)  # Optional, unquoted_string
            # p_location = item.get("location", None)  # Optional, geo_value
            # p_attributes = item.get("attributes", None)  # Optional, key_value_list
            # if p_attributes:
            #     try:
            #         attributes = dictionary_from_key_value_list(p_attributes, glb_idx)
            #     except Exception as e:
            #         issues.append(Issue(itype=3,
            #                             description=str(e),
            #                             location=IssueLocation(sheet_name=command_name, row=row, column=None)))
            #         return
            # else:
            #     attributes = {}
            #

            # Process specific fields

            # Obtain the parent: it must exist. It could be created dynamically but it's important to specify attributes
            parent_processor = None
            field_value = fields_dict.get("parent_processor", None)
            if field_value:
                parent_processor = find_processor_by_name(state=glb_idx, processor_name=field_value)
                if not parent_processor:
                    issues.append(create_issue(3, f"Specified parent processor, '{field_value}', does not exist"))
                    return

            # Find or create processor and REGISTER it in "glb_idx"
            # TODO Now, only Simple name allowed
            # TODO Improve allowing hierarchical names, and hierarchical names with wildcards
            # TODO Improve allowing CLONE(<processor name>)
            # TODO Pass the attributes:
            # TODO p_type, p_f_or_s, p_i_or_a, p_alias, p_description, p_copy_interfaces
            if fields_dict.get("clone_processor", None):
                # TODO Find origin processor
                # TODO Clone it
                pass
            else:
                processor_attributes = {f.name: fields_dict[f.name] for f in commands["Processors"] if f.attribute_of == Processor}

                # If key "attributes" exist, expand it.
                # E.g. {"a": 1, "b": 2, "attributes": {"c": 3, "d": 4}} -> {'a': 1, 'b': 2, 'c': 3, 'd': 4}
                if "attributes" in processor_attributes:
                    field_value = processor_attributes["attributes"]
                    del processor_attributes["attributes"]
                    if field_value:
                        processor_attributes.update(field_value)

                p = find_or_create_processor(
                    state=glb_idx,
                    name=fields_dict["processor"],
                    proc_attributes=processor_attributes,
                    proc_location=Geolocation.create(fields_dict["geolocation_ref"], fields_dict["geolocation_code"])
                )

            # Add to ProcessorsGroup, if specified
            field_value = fields_dict.get("processor_group", None)
            if field_value:
                p_set = p_sets.get(field_value, ProcessorsSet(field_value))
                p_sets[field_value] = p_set
                if p_set.append(p, glb_idx):  # Appends codes to the pset if the processor was not member of the pset
                    p_set.append_attributes_codes(fields_dict["attributes"])

            # Add Relationship "part-of" if parent was specified
            # The processor may have previously other parent processors that will keep its parentship
            if parent_processor:
                # Create "part-of" relationship
                o1 = ProcessorsRelationPartOfObservation.create_and_append(parent_processor, p, None)  # Part-of
                glb_idx.put(o1.key(), o1)

        def create_issue(itype: int, description: str) -> Issue:
            return Issue(itype=itype,
                         description=description,
                         location=IssueLocation(sheet_name=command_name, row=row, column=None))

        issues = []
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        command_name = self._content["command_name"]

        # CommandField definitions for the fields of Interface command
        fields = {f.name: f for f in commands["Processors"]}
        # Obtain the names of all parameters
        parameters = [p.name for p in glb_idx.get(Parameter.partial_key())]
        # Obtain the names of all processors
        all_processors = get_processor_names_to_processors_dictionary(glb_idx)

        # Process parsed information
        for line in self._content["items"]:
            row = line["_row"]
            for sub_line in parse_and_unfold_line(line):
                process_line(sub_line)

        return issues, None  # Issues, Output

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