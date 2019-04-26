import re
from typing import NoReturn, Dict, Any

from backend import IssuesOutputPairType
from backend.command_executors import CommandExecutionError, BasicCommand
from backend.command_executors.execution_helpers import parse_line, classify_variables, \
    obtain_dictionary_with_literal_fields
from backend.command_field_definitions import get_command_fields_from_class
from backend.command_generators import IType
from backend.command_generators.parser_field_parsers import string_to_ast, processor_names
from backend.common.helper import strcmp
from backend.model_services import get_case_study_registry_objects
from backend.models.musiasem_concepts import Factor, RelationClassType, Parameter, Processor, FactorType, \
    FactorTypesConverter
from backend.models.musiasem_concepts_helper import create_relation_observations, find_factor_types_transform_relation
from backend.solving import get_processor_names_to_processors_dictionary


class RelationshipsCommand(BasicCommand):
    def __init__(self, name: str):
        BasicCommand.__init__(self, name, get_command_fields_from_class(self.__class__))

    def execute(self, state: "State") -> IssuesOutputPairType:
        self._issues = []

        self._glb_idx, _, hh, datasets, _ = get_case_study_registry_objects(state)

        # Obtain the names of all parameters
        parameters = [p.name for p in self._glb_idx.get(Parameter.partial_key())]

        # Obtain the names of all processors
        all_processors = get_processor_names_to_processors_dictionary(self._glb_idx)

        for row in self._content["items"]:
            for sub_row in self.parse_and_unfold_line(row, hh, datasets, parameters, all_processors):
                try:
                    self._init_and_process_row(sub_row)
                except CommandExecutionError as e:
                    self._add_issue(IType.ERROR, str(e))

        return self._issues, None

    def _process_row(self, row: Dict[str, Any]) -> NoReturn:
        # source_cardinality = self._fields_values["source_cardinality"]
        # target_cardinality = self._fields_values["target_cardinality"]

        source_processor = self._get_processor_from_field("source_processor")
        target_processor = self._get_processor_from_field("target_processor")
        attributes = self._get_attributes_from_field("attributes")

        try:  # Get relation class type
            relation_class = RelationClassType.from_str(self._fields_values["relation_type"])
        except NotImplementedError as e:
            raise CommandExecutionError(str(e))

        # Allow use of column "BackInterface" for "Flow" relation type
        if self._fields_values["back_interface"] is not None and relation_class == RelationClassType.ff_directed_flow:
            relation_class = RelationClassType.ff_directed_flow_back

        if relation_class.is_between_processors:
            create_relation_observations(self._glb_idx, source_processor, [(target_processor, relation_class)],
                                         relation_class, None, attributes=attributes)

        elif relation_class.is_between_interfaces:
            # Processors should be the same when relation is "Scale Change"
            if relation_class == RelationClassType.ff_scale_change and source_processor.name != target_processor.name:
                raise CommandExecutionError(
                    f"Source and target processors should be the same for a relation of type"
                    f" 'Scale Change': '{source_processor.name}' != '{target_processor.name}'")

            # Look for interfaces
            source_interface = self._get_interface_from_field("source_interface", source_processor)
            target_interface = self._get_interface_from_field("target_interface", target_processor)

            if relation_class == RelationClassType.ff_directed_flow_back:
                back_interface = self._get_interface_from_field("back_interface", source_processor)

                if back_interface.taxon != target_interface.taxon:
                    raise CommandExecutionError(f"The type of target and back interfaces should be the same. Target: "
                                                f"{target_interface.taxon.name}; Back: {back_interface.taxon.name}")

                attributes.update(dict(back_interface=back_interface))

            if relation_class.is_flow:
                # Check for correct interfaces orientation (input/output) of source and target
                self._check_flow_orientation(
                    source_processor, target_processor, source_interface, target_interface,
                    is_direct_flow=(relation_class == RelationClassType.ff_directed_flow)
                )

            if source_interface.taxon != target_interface.taxon:
                attributes.update(dict(interface_types_converter=self._get_interface_types_converter(
                    source_interface.taxon, source_processor, target_interface.taxon, target_processor))
                )

            create_relation_observations(self._glb_idx, source_interface,
                                         [(target_interface, relation_class, self._fields_values["flow_weight"])],
                                         relation_class, None, attributes=attributes)

    def _get_interface_types_converter(
            self,
            source_interface_type: FactorType, source_processor: Processor,
            target_interface_type: FactorType, target_processor: Processor) -> FactorTypesConverter:
        """Check if a transformation between interfaces has been specified"""

        interface_types_transforms = find_factor_types_transform_relation(
            self._glb_idx, source_interface_type, target_interface_type, source_processor, target_processor)

        if len(interface_types_transforms) == 0:
            raise CommandExecutionError(f"Interface types are not the same (and transformation from one "
                                        f"to the other cannot be performed). Origin: "
                                        f"{source_interface_type.name}; Target: {target_interface_type.name}")
        elif len(interface_types_transforms) > 1:
            raise CommandExecutionError(
                f"Multiple transformations can be applied between interfaces. Origin: "
                f"{source_interface_type.name}; Target: {target_interface_type.name}")

        return interface_types_transforms[0].converter

    def _check_flow_orientation(self, source_processor: Processor, target_processor: Processor,
                                source_interface: Factor, target_interface: Factor, is_direct_flow: bool):
        allowed_source_orientation = ("Output" if is_direct_flow else "Input")

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

            if child_processor not in parent_processor.children(self._glb_idx):
                raise CommandExecutionError(f"The processor '{child_processor.name}' should be part of the "
                                            f"processor '{parent_processor.name}' when using the same interface "
                                            f"orientation '{source_interface.orientation}'.")

        else:  # Orientations are different
            if not strcmp(source_interface.orientation, allowed_source_orientation):
                raise CommandExecutionError(f"The source interface '{source_interface.full_name}' has the wrong "
                                            f"orientation '{source_interface.orientation}'.")

            if strcmp(target_interface.orientation, allowed_source_orientation):
                raise CommandExecutionError(f"The target interface '{target_interface.full_name}' has the wrong "
                                            f"orientation '{target_interface.orientation}'.")

    def parse_and_unfold_line(self, item, hh, datasets, parameters, all_processors):
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
        # Look for multiple items in r_source_processor_name, source_interface_name,
        #                            r_target_processor_name, target_interface_name
        if item["_complex"]:
            asts = parse_line(item, self._fields_values)
            if item["_expandable"]:
                # It is an expandable line
                # Look for fields which are specified to be variable in order to originate the expansion
                res = classify_variables(asts, datasets, hh, parameters)
                ds_list = res["datasets"]
                ds_concepts = res["ds_concepts"]
                h_list = res["hierarchies"]
                if len(ds_list) >= 1 and len(h_list) >= 1:
                    self._add_issue(IType.ERROR, "Dataset(s): "+", ".join([d.name for d in ds_list])+", and hierarchy(ies): "+", ".join([h.name for h in h_list])+", have been specified. Either a single dataset or a single hiearchy is supported.")
                    return
                elif len(ds_list) > 1:
                    self._add_issue(IType.ERROR, "More than one dataset has been specified: "+", ".join([d.name for d in ds_list])+", just one dataset is supported.")
                    return
                elif len(h_list) > 1:
                    self._add_issue(IType.ERROR, "More than one hierarchy has been specified: " + ", ".join([h.name for h in h_list])+", just one hierarchy is supported.")
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
                        self._add_issue(IType.ERROR, "It is not possible to use a measure if not all dimensions are used (cannot assume implicit aggregation)")
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
                        for f in self._fields_values:
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
