import json

from backend import CommandField
from backend.command_field_definitions import processor_types, functional_or_structural, instance_or_archetype, \
    copy_interfaces_mode
from backend.command_generators import Issue
from backend.command_generators.parser_ast_evaluators import dictionary_from_key_value_list
from backend.command_generators.spreadsheet_command_parsers_v2 import IssueLocation
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts import ProcessorsSet, ProcessorsRelationPartOfObservation
from backend.models.musiasem_concepts_helper import find_or_create_processor, find_processor_by_name


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

        def process_line(item):
            # Read variables
            p_name = item.get("processor", None)  # Mandatory, simple_ident
            p_group = item.get("processor_group", None)  # Optional, simple_ident
            p_type = item.get("processor_type", processor_types[0])  # Optional, simple_ident
            p_f_or_s = item.get("functional_or_structural", functional_or_structural[0])  # Optional, simple_ident
            p_copy_interfaces = item.get("copy_interfaces_mode", copy_interfaces_mode[0])
            p_i_or_a = item.get("instance_or_archetype", instance_or_archetype[0])  # Optional, simple_ident
            p_parent = item.get("parent_processor", None)  # Optional, simple_ident
            p_clone_processor = item.get("clone_processor", None)  # Optional, simple_ident
            p_alias = item.get("alias", None)  # Optional, simple_ident
            p_description = item.get("description", None)  # Optional, unquoted_string
            p_location = item.get("location", None)  # Optional, geo_value
            p_attributes = item.get("attributes", None)  # Optional, key_value_list
            if p_attributes:
                try:
                    attributes = dictionary_from_key_value_list(p_attributes, glb_idx)
                except Exception as e:
                    issues.append(Issue(itype=3,
                                        description=str(e),
                                        location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                    return
            else:
                attributes = {}

            # Process variables
            if not p_name:
                issues.append(Issue(itype=3,
                                    description="Empty processor name. Skipped.",
                                    location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                return

            parent_processor = None
            if p_parent:
                # Obtain the parent
                # It must exist (it could be created dynamically, but it is important to specify attributes)
                parent_processor = find_processor_by_name(state=glb_idx, processor_name=p_parent)
                if not parent_processor:
                    issues.append(Issue(itype=3,
                                        description="Specified parent processor, '"+p_parent+"', does not exist",
                                        location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                    return

            is_context_processor = True

            # TODO
            # If it is a context processor, create it as "local"
            # Create accompanying "local-environment"
            # Create non-local and non-local-environment

            # Find or create processor and REGISTER it in "glb_idx"
            # TODO Now, only Simple name allowed
            # TODO Improve allowing hierarchical names, and hierarchical names with wildcards
            # TODO Improve allowing CLONE(<processor name>)
            # TODO Pass the attributes:
            # TODO p_type, p_f_or_s, p_i_or_a, p_alias, p_description, p_copy_interfaces
            if p_clone_processor:
                # TODO Find origin processor
                # TODO Clone it
                pass
            else:
                p = find_or_create_processor(state=glb_idx,
                                             name=p_name,
                                             proc_external=None,
                                             proc_attributes=attributes,
                                             proc_location=p_location)

            # Add to ProcessorsGroup, if specified
            if p_group:
                if p_group not in p_sets:
                    p_sets[p_group] = set()
                p_sets[p_group].add(p)

            # Add Relationship "part-of" if parent was specified
            # The processor may have previously other parent processors that will keep its parentship
            if parent_processor:
                # Create "part-of" relationship
                o1 = ProcessorsRelationPartOfObservation.create_and_append(parent_processor, p, None)  # Part-of
                glb_idx.put(o1.key(), o1)

        issues = []
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        name = self._content["command_name"]

        # Process parsed information
        for r, line in enumerate(self._content["items"]):
            # If the line contains a reference to a dataset or hierarchy, expand it
            # If not, process it directly
            is_expansion = False
            if is_expansion:
                # TODO Iterate through dataset and/or hierarchy elements, producing a list of new items
                pass
            else:
                process_line(line)

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