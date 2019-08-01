import json
import re
from typing import Dict, Any

from backend import CommandField
from backend.command_executors import BasicCommand, CommandExecutionError
from backend.command_executors.execution_helpers import parse_line, classify_variables, \
    obtain_dictionary_with_literal_fields
from backend.command_executors.version2.relationships_command import obtain_matching_processors
from backend.command_field_definitions import get_command_fields_from_class
from backend.command_generators import Issue, IssueLocation, IType
from backend.command_generators.parser_ast_evaluators import dictionary_from_key_value_list
from backend.command_generators.parser_field_parsers import string_to_ast, processor_names
from backend.common.helper import head, strcmp, ifnull
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


class ProcessorsCommand(BasicCommand):
    def __init__(self, name: str):
        BasicCommand.__init__(self, name, get_command_fields_from_class(self.__class__))

    # def _execute(self, state: "State"):
    #     """
    #     Create empty Processors, potentially related with PartOf relationship
    #
    #     The Name of a new Processor can be:
    #     * Simple name
    #     * Hierarchical name
    #     * Clone previously declared processor (and children), where "previously declared processor" can be simple or complex
    #
    #     If name is hierarchic, a formerly declared processor is assumed.
    #     If a parent processor is specified, the processor will be related to the new parent (if parents are different -> multiple functionality)
    #     In a hierarchy, children inherit interfaces from parents
    #                     if parents do not have interfaces, do they "receive" interfaces from immediate child?
    #                     - Only if children have them already (separate copy)
    #                     - If children do not have them already (connected copy)
    #
    #     If CLONE(<existing processor>) is specified, a copy of the <existing processor> (and children) will be created.
    #     In this case, the Parent field is mandatory (if not, no unique name will be available). For a pair "parent processor - clone processor", this operation is unique (unless, a naming syntax for the copy is invented)
    #     If child processors had interface, copy them into the parent processor (the default move is downwards, from parent to child)
    #
    #     Later, when interfaces are attached to processors,
    #     :param state:
    #     :return:
    #     """
    #     def parse_and_unfold_line(item):
    #         # Consider multiplicity because of:
    #         # - A dataset (only one). First a list of dataset concepts used in the line is obtained.
    #         #   Then the unique tuples formed by them are obtained.
    #         # - Processor name.
    #         #   - A set of processors (wildcard or filter by attributes)
    #         #   - A set of interfaces (according to another filter?)
    #         # - Multiple types of relation
    #         # - Both (first each dataset record applied -expanded-, then the name evaluation is applied)
    #         # - UNRESOLVED: expressions are resolved partially. Parts where parameters
    #         # expressions depending on parameters. Only the part of the expression depending on varying things
    #         # - The processor name could be a concatenation of multiple literals
    #         #
    #         # Look for multiple items in r_source_processor_name, r_source_interface_name,
    #         #                            r_target_processor_name, r_target_interface_name
    #         if item["_complex"]:
    #             asts = parse_line(item, fields)
    #             if item["_expandable"]:
    #                 # It is an expandable line
    #                 # Look for fields which are specified to be variable in order to originate the expansion
    #                 res = classify_variables(asts, datasets, hh, parameters)
    #                 ds_list = res["datasets"]
    #                 ds_concepts = res["ds_concepts"]
    #                 h_list = res["hierarchies"]
    #                 if len(ds_list) >= 1 and len(h_list) >= 1:
    #                     issues.append(create_issue(IType.ERROR, "Dataset(s): "+", ".join([d.name for d in ds_list])+", and hierarchy(ies): "+", ".join([h.name for h in h_list])+", have been specified. Either a single dataset or a single hiearchy is supported."))
    #                     return
    #                 elif len(ds_list) > 1:
    #                     issues.append(create_issue(IType.ERROR, "More than one dataset has been specified: "+", ".join([d.name for d in ds_list])+", just one dataset is supported."))
    #                     return
    #                 elif len(h_list) > 1:
    #                     issues.append(create_issue(IType.ERROR, "More than one hierarchy has been specified: " + ", ".join([h.name for h in h_list])+", just one hierarchy is supported."))
    #                     return
    #                 const_dict = obtain_dictionary_with_literal_fields(item, asts)
    #                 if len(ds_list) == 1:
    #                     # If a measure is requested and not all dimensions are used, aggregate or
    #                     # issue an error (because it is not possible to reduce without aggregation).
    #                     # If only dimensions are used, then obtain all the unique tuples
    #                     ds = ds_list[0]
    #                     measure_requested = False
    #                     all_dimensions = set([c.code for c in ds.dimensions if not c.is_measure])
    #                     for con in ds_concepts:
    #                         for c in ds.dimensions:
    #                             if strcmp(c.code, con):
    #                                 if c.is_measure:
    #                                     measure_requested = True
    #                                 else:  # Dimension
    #                                     all_dimensions.remove(c.code)
    #                     only_dimensions_requested = len(all_dimensions) == 0
    #
    #                     if measure_requested and not only_dimensions_requested:
    #                         issues.append(create_issue(IType.ERROR, "It is not possible to use a measure if not all dimensions are used (cannot assume implicit aggregation)"))
    #                         return
    #                     elif not measure_requested and not only_dimensions_requested:
    #                         # TODO Reduce the dataset to the unique tuples (consider the current case -sensitive or not-sensitive-)
    #                         data = None
    #                     else:  # Take the dataset as-is
    #                         data = ds.data
    #
    #                     for row in data.iterrows():
    #                         item2 = const_dict.copy()
    #
    #                         d = {}
    #                         for c in ds_concepts:
    #                             d["{" + ds.code + "." + c + "}"] = row[c]
    #                         # Expand in all fields
    #                         for f in fields:
    #                             if f not in const_dict:
    #                                 # Replace all
    #                                 string = item[f]
    #                                 # TODO Could iterate through the variables in the field (not IN ALL FIELDS of the row)
    #                                 for item in sorted(d.keys(), key=len, reverse=True):
    #                                     string = re.sub(item, d[item], string)
    #                                 item2[f] = string
    #                         # Now, look for wildcards where it is allowed
    #                         r_source_processor_name = string_to_ast(processor_names, item2.get("source_processor", None))
    #                         r_target_processor_name = string_to_ast(processor_names, item2.get("target_processor", None))
    #                         if ".." in r_source_processor_name or ".." in r_target_processor_name:
    #                             if ".." in r_source_processor_name:
    #                                 source_processor_names = obtain_matching_processors(r_source_processor_name, all_processors)
    #                             else:
    #                                 source_processor_names = [r_source_processor_name]
    #                             if ".." in r_target_processor_name:
    #                                 target_processor_names = obtain_matching_processors(r_target_processor_name, all_processors)
    #                             else:
    #                                 target_processor_names = [r_target_processor_name]
    #                             for s in source_processor_names:
    #                                 for t in target_processor_names:
    #                                     item3 = item2.copy()
    #                                     item3["source_processor"] = s
    #                                     item3["target_processor"] = t
    #                                     print("Multiple by dataset and wildcard: " + str(item3))
    #                                     yield item3
    #                         else:
    #                             print("Multiple by dataset: " + str(item3))
    #                             yield item2
    #                 elif len(h_list) == 1:
    #                     pass
    #                 else:  # No dataset, no hierarchy of categories, but still complex, because of wildcards
    #                     wildcard_in_source = ".." in item.get("source_processor", "")
    #                     wildcard_in_target = ".." in item.get("target_processor", "")
    #                     if wildcard_in_source or wildcard_in_target:
    #                         r_source_processor_name = string_to_ast(processor_names, item.get("source_processor", None))
    #                         r_target_processor_name = string_to_ast(processor_names, item.get("target_processor", None))
    #                         if wildcard_in_source:
    #                             source_processor_names = obtain_matching_processors(r_source_processor_name, all_processors)
    #                         else:
    #                             source_processor_names = [item["source_processor"]]
    #                         if wildcard_in_target:
    #                             target_processor_names = obtain_matching_processors(r_target_processor_name, all_processors)
    #                         else:
    #                             target_processor_names = [item["target_processor"]]
    #                         for s in source_processor_names:
    #                             for t in target_processor_names:
    #                                 item3 = const_dict.copy()
    #                                 item3["source_processor"] = s
    #                                 item3["target_processor"] = t
    #                                 print("Multiple by wildcard: "+str(item3))
    #                                 yield item3
    #                     else:
    #                         # yield item
    #                         raise Exception("If 'complex' is signaled, it should not pass by this line")
    #         else:
    #             # print("Single: "+str(item))
    #             yield item
    #
    #     issues = []
    #     glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
    #     command_name = self._content["command_name"]
    #
    #     # CommandField definitions for the fields of Interface command
    #     fields: Dict[str, CommandField] = {f.name: f for f in get_command_fields_from_class(self.__class__)}
    #     # Obtain the names of all parameters
    #     parameters = [p.name for p in glb_idx.get(Parameter.partial_key())]
    #     # Obtain the names of all processors
    #     all_processors = get_processor_names_to_processors_dictionary(glb_idx)
    #
    #     # Process parsed information
    #     for line in self._content["items"]:
    #         row = line["_row"]
    #         for sub_line in parse_and_unfold_line(line):
    #             self._process_row(sub_line)
    #
    #     return issues, None  # Issues, Output

    def _process_row(self, field_values: Dict[str, Any]) -> None:

        # Transform text of "attributes" into a dictionary
        if field_values.get("attributes"):
            try:
                field_values["attributes"] = dictionary_from_key_value_list(field_values["attributes"], self._glb_idx)
            except Exception as e:
                self._add_issue(IType.ERROR, str(e))
                return
        else:
            field_values["attributes"] = {}

        # Process specific fields

        # Obtain the parent: it must exist. It could be created dynamically but it's important to specify attributes
        if field_values.get("parent_processor"):
            try:
                parent_processor = self._get_processor_from_field("parent_processor")
            except CommandExecutionError:
                self._add_issue(IType.ERROR, f"Specified parent processor, '{field_values.get('parent_processor')}', does not exist")
                return
        else:
            parent_processor = None

        # Find or create processor and REGISTER it in "glb_idx"
        # TODO Now, only Simple name allowed
        # TODO Improve allowing hierarchical names, and hierarchical names with wildcards
        # TODO Improve allowing CLONE(<processor name>)
        # TODO Pass the attributes:
        #  p_type, p_f_or_s, p_i_or_a, p_alias, p_description, p_copy_interfaces
        if field_values.get("clone_processor"):
            # TODO Find origin processor
            # TODO Clone it
            pass
        else:
            # Get internal and user-defined attributes in one dictionary
            attributes = {c.name: field_values[c.name] for c in self._command_fields if c.attribute_of == Processor}
            attributes.update(field_values["attributes"])

            if not attributes.get("processor_system"):
                attributes["processor_system"] = "default"

            # Needed to support the new name of the field, "Accounted" (previously it was "InstanceOrArchetype")
            # (internally the values have the same meaning, "Instance" for a processor which has to be accounted,
            # "Archetype" for a processor which hasn't)
            v = attributes.get("instance_or_archetype", None)
            if strcmp(v, "Yes"):
                v = "Instance"
            elif strcmp(v, "No"):
                v = "Archetype"
            if v:
                attributes["instance_or_archetype"] = v

            p = find_or_create_processor(
                state=self._glb_idx,
                name=field_values["processor"],  # TODO: add parent hierarchical name
                proc_attributes=attributes,
                proc_location=Geolocation.create(field_values["geolocation_ref"], field_values["geolocation_code"])
            )

        # Add to ProcessorsGroup, if specified
        field_val = field_values.get("processor_group")
        if field_val:
            p_set = self._p_sets.get(field_val, ProcessorsSet(field_val))
            self._p_sets[field_val] = p_set
            if p_set.append(p, self._glb_idx):  # Appends codes to the pset if the processor was not member of the pset
                p_set.append_attributes_codes(field_values["attributes"])

        # Add Relationship "part-of" if parent was specified
        # The processor may have previously other parent processors that will keep its parentship
        if parent_processor:
            # Create "part-of" relationship
            o1 = ProcessorsRelationPartOfObservation.create_and_append(parent_processor, p, None)  # Part-of
            self._glb_idx.put(o1.key(), o1)
