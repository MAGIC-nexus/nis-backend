import json
import re

from pint import UndefinedUnitError
from typing import Dict, Any

from nexinfosys import ureg, CommandField
from nexinfosys.command_executors import BasicCommand
from nexinfosys.command_executors.execution_helpers import parse_line, classify_variables, \
    obtain_dictionary_with_literal_fields
from nexinfosys.command_generators import parser_field_parsers, Issue, IssueLocation, IType
from nexinfosys.command_generators.parser_ast_evaluators import dictionary_from_key_value_list, ast_to_string
from nexinfosys.common.helper import strcmp, first, ifnull
from nexinfosys.model_services import IExecutableCommand, get_case_study_registry_objects
from nexinfosys.models.musiasem_concepts import PedigreeMatrix, Reference, FactorType, \
    Processor, Factor, FactorInProcessorType, Observer, Parameter, GeographicReference, ProvenanceReference, \
    BibliographicReference
from nexinfosys.models.musiasem_concepts_helper import _create_or_append_quantitative_observation
from nexinfosys.solving import get_processor_names_to_processors_dictionary
from nexinfosys.command_field_definitions import get_command_fields_from_class


class InterfacesAndQualifiedQuantitiesCommand(BasicCommand):
    def __init__(self, name: str):
        BasicCommand.__init__(self, name, get_command_fields_from_class(self.__class__))

    def _process_row(self, field_values: Dict[str, Any]) -> None:
        """
        Process a dictionary representing a row of the Interfaces command. The dictionary can come directly from
        the worksheet or from a dataset.

        :param field_values: dictionary
        """

        # Interface
        f_processor_name = field_values.get("processor")
        f_interface_type_name = field_values.get("interface_type")
        f_interface_name = field_values.get("interface")  # A "simple_ident", optional
        f_location = field_values.get("location")
        f_orientation = field_values.get("orientation")
        # f_roegen_type = fields_value.get("roegen_type")
        # f_sphere = fields_value.get("sphere")
        # f_opposite_processor_type = fields_value.get("opposite_processor_type")
        # f_geolocation_ref = fields_value.get("geolocation_ref")
        # f_geolocation_code = fields_value.get("geolocation_code")

        # Qualified Quantity
        f_value = field_values.get("value")
        f_unit = field_values.get("unit")
        f_uncertainty = field_values.get("uncertainty")
        f_assessment = field_values.get("assessment")
        f_pedigree_matrix = field_values.get("pedigree_matrix")
        f_pedigree = field_values.get("pedigree")
        f_relative_to = field_values.get("relative_to")
        f_time = field_values.get("time")
        f_source = field_values.get("qq_source")
        f_number_attributes = field_values.get("number_attributes", {})
        f_comments = field_values.get("comments")

        # Transform text of "interface_attributes" into a dictionary
        field_val = field_values.get("interface_attributes")
        if field_val:
            try:
                field_values["interface_attributes"] = dictionary_from_key_value_list(field_val, self._glb_idx)
            except Exception as e:
                self._add_issue(IType.ERROR, str(e))
                return
        else:
            field_values["interface_attributes"] = {}

        # Transform text of "number_attributes" into a dictionary
        if f_number_attributes:
            try:
                number_attributes = dictionary_from_key_value_list(f_number_attributes, self._glb_idx)
            except Exception as e:
                self._add_issue(IType.ERROR, str(e))
                return
        else:
            number_attributes = {}

        # f_processor_name -> p
        # f_interface_type_name -> it
        # f_interface_name -> i
        #
        # IF NOT i AND it AND p => i_name = it.name => get or create "i"
        # IF i AND it AND p => get or create "i", IF "i" exists, i.it MUST BE equal to "it" (IF NOT, error)
        # IF i AND p AND NOT it => get "i" (MUST EXIST)
        if not f_interface_name:
            if not f_interface_type_name:
                self._add_issue(IType.ERROR, "At least one of InterfaceType or Interface must be defined")
                return

            possibly_local_interface_name = None
            f_interface_name = f_interface_type_name
        else:
            possibly_local_interface_name = f_interface_name

        # Check existence of PedigreeMatrix, if used
        if f_pedigree_matrix and f_pedigree:
            pm = self._glb_idx.get(PedigreeMatrix.partial_key(name=f_pedigree_matrix))
            if len(pm) == 0:
                self._add_issue(IType.ERROR, "Could not find Pedigree Matrix '" + f_pedigree_matrix + "'")
                return
            else:
                try:
                    lst = pm[0].get_modes_for_code(f_pedigree)
                except:
                    self._add_issue(IType.ERROR, "Could not decode Pedigree '" + f_pedigree + "' for Pedigree Matrix '" + f_pedigree_matrix + "'")
                    return
        elif f_pedigree and not f_pedigree_matrix:
            self._add_issue(IType.ERROR, "Pedigree specified without accompanying Pedigree Matrix")
            return

        # Source
        if f_source:
            try:
                ast = parser_field_parsers.string_to_ast(parser_field_parsers.reference, f_source)
                ref_id = ast["ref_id"]
                references = self._glb_idx.get(ProvenanceReference.partial_key(ref_id))
                if len(references) == 1:
                    source = references[0]
                else:
                    references = self._glb_idx.get(BibliographicReference.partial_key(ref_id))
                    if len(references) == 1:
                        source = references[0]
                    else:
                        self._add_issue(IType.ERROR, f"Reference '{f_source}' not found")
            except:
                # TODO Change when Ref* are implemented
                source = f_source + " (not found)"
        else:
            source = None

        # Geolocation
        if f_location:
            try:
                # TODO Change to parser for Location (includes references, but also Codes)
                ast = parser_field_parsers.string_to_ast(parser_field_parsers.reference, f_location)
                ref_id = ast["ref_id"]
                references = self._glb_idx.get(GeographicReference.partial_key(ref_id))
                if len(references) == 1:
                    geolocation = references[0]
            except:
                geolocation = f_location
        else:
            geolocation = None

        # Find Processor
        # TODO Allow creating a basic Processor if it is not found?
        p = self._glb_idx.get(Processor.partial_key(f_processor_name))
        if len(p) == 0:
            self._add_issue(IType.ERROR, "Processor '" + f_processor_name + "' not declared previously")
            return
        elif len(p) > 1:
            self._add_issue(IType.ERROR, "Processor '" + f_processor_name + "' found " + str(len(p)) + " times. It must be uniquely identified.")
            return
        else:
            p = p[0]

        # Try to find Interface
        ft: FactorType = None
        f = self._glb_idx.get(Factor.partial_key(processor=p, name=f_interface_name))
        if len(f) == 1:
            f: Factor = f[0]
            ft: FactorType = f.taxon
            if f_interface_type_name:
                if not strcmp(ft.name, f_interface_type_name):
                    self._add_issue(IType.WARNING, f"The InterfaceType of the Interface, {ft.name} "
                                    f"is different from the specified InterfaceType, {f_interface_type_name}. Record skipped.")
                    return
        elif len(f) > 1:
            self._add_issue(IType.ERROR, f"Interface '{f_interface_name}' found {str(len(f))} times. "
                                         f"It must be uniquely identified.")
            return
        elif len(f) == 0:
            f: Factor = None  # Does not exist, create it below
            if not f_orientation:
                self._add_issue(IType.ERROR, f"Orientation must be defined for new Interfaces")
                return

        # InterfaceType still not found
        if not ft:
            # Find FactorType
            # TODO Allow creating a basic FactorType if it is not found
            ft = self._glb_idx.get(FactorType.partial_key(f_interface_type_name))
            if len(ft) == 0:
                self._add_issue(IType.ERROR, f"InterfaceType '{f_interface_type_name}' not declared previously")
                return
            elif len(ft) > 1:
                self._add_issue(IType.ERROR, f"InterfaceType '{f_interface_type_name}' found {str(len(ft))} times. "
                                       f"It must be uniquely identified.")
                return
            else:
                ft = ft[0]

        # Get attributes default values taken from Interface Type or Processor attributes
        default_values = {
            "sphere": ft.sphere,
            "roegen_type": ft.roegen_type,
            "opposite_processor_type": ft.opposite_processor_type
        }

        # Get internal and user-defined attributes in one dictionary
        attributes = {c.name: ifnull(field_values[c.name], default_values.get(c.name))
                      for c in self._command_fields if c.attribute_of == Factor}

        if not f:
            attributes.update(field_values["interface_attributes"])

            f = Factor.create_and_append(f_interface_name,
                                         p,
                                         in_processor_type=FactorInProcessorType(
                                             external=False,
                                             incoming=False
                                         ),
                                         taxon=ft,
                                         geolocation=f_location,
                                         tags=None,
                                         attributes=attributes)
            self._glb_idx.put(f.key(), f)

        elif not f.compare_attributes(attributes):
            self._add_issue(IType.ERROR, f"The same interface is being redeclared with different properties.")

        # Find Observer
        oer = self._glb_idx.get(Observer.partial_key(f_source))
        if not oer:
            self._add_issue(IType.WARNING, f"Observer '{f_source}' has not been found.")
        else:
            oer = oer[0]

        if f_relative_to:
            ast = parser_field_parsers.string_to_ast(parser_field_parsers.factor_unit, f_relative_to)
            relative_to_interface_name = ast_to_string(ast["factor"])

            rel_unit_name = ast["unparsed_unit"]
            try:
                f_unit = str((ureg(f_unit) / ureg(rel_unit_name)).units)
            except (UndefinedUnitError, AttributeError) as ex:
                self._add_issue(IType.ERROR, f"The final unit could not be computed, interface '{f_unit}' / "
                                       f"relative_to '{rel_unit_name}': {str(ex)}")
                return

            f_relative_to = first(f.processor.factors, lambda ifc: strcmp(ifc.name, relative_to_interface_name))

            if not f_relative_to:
                self._add_issue(IType.ERROR, f"Interface specified in 'relative_to' column "
                                       f"'{relative_to_interface_name}' has not been found.")
                return

        if f_value is None and f_relative_to is not None:
            f_value = "0"
            self._add_issue(IType.WARNING, f"Field 'value' should be defined for unitary processors, i.e. those having a "
                                     f"'relative_to' interface. Using value '0'.")

        # Create quantitative observation
        if f_value is not None:
            # If an observation exists then "time" is mandatory
            if not f_time:
                self._add_issue(IType.ERROR, f"Field 'time' needs to be specified for the given observation.")
                return

            o = _create_or_append_quantitative_observation(f,
                                                           f_value, f_unit, f_uncertainty, f_assessment, f_pedigree, f_pedigree_matrix,
                                                           oer,
                                                           f_relative_to,
                                                           f_time,
                                                           None,
                                                           f_comments,
                                                           None, number_attributes
                                                           )

            # TODO Register? Disable for now. Observation can be obtained from a pass over all Interfaces
            # glb_idx.put(o.key(), o)
