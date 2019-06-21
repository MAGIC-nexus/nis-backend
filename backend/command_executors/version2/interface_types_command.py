import json

from backend.command_generators import Issue, IssueLocation, IType
from backend.command_generators.parser_ast_evaluators import dictionary_from_key_value_list
from backend.common.helper import strcmp
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts import Hierarchy, FactorType, FlowFundRoegenType


class InterfaceTypesCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        def process_line(item):
            # Read variables
            ft_h_name = item.get("interface_type_hierarchy", "_default")  # "_default" InterfaceType Hierarchy NAME <<<<<<
            ft_name = item.get("interface_type")
            ft_sphere = item.get("sphere")
            ft_roegen_type = item.get("roegen_type")
            ft_parent = item.get("parent_interface_type")
            ft_formula = item.get("formula")
            ft_description = item.get("description")
            ft_unit = item.get("unit")
            ft_opposite_processor_type = item.get("opposite_processor_type")
            ft_attributes = item.get("attributes", {})
            print(str(type(ft_attributes)))
            if ft_attributes:
                try:
                    ft_attributes = dictionary_from_key_value_list(ft_attributes, glb_idx)
                except Exception as e:
                    issues.append(Issue(itype=IType.ERROR,
                                        description=str(e),
                                        location=IssueLocation(sheet_name=name, row=r, column=None)))
                    return
            else:
                ft_attributes = {}

            # Process
            # Mandatory fields
            if not ft_h_name:
                issues.append(Issue(itype=IType.ERROR,
                                    description="Empty interface type hierarchy name. Skipped.",
                                    location=IssueLocation(sheet_name=name, row=r, column=None)))
                return

            if not ft_name:
                issues.append(Issue(itype=IType.ERROR,
                                    description="Empty interface type name. Skipped.",
                                    location=IssueLocation(sheet_name=name, row=r, column=None)))
                return

            # Check if a hierarchy of interface types by the name <ft_h_name> exists, if not, create it and register it
            hie = glb_idx.get(Hierarchy.partial_key(name=ft_h_name))
            if not hie:
                hie = Hierarchy(name=ft_h_name, type_name="interfacetype")
                glb_idx.put(hie.key(), hie)
            else:
                hie = hie[0]

            # If parent defined, check if it exists
            # (it must be registered both in the global registry AND in the hierarchy)
            if ft_parent:
                parent = glb_idx.get(FactorType.partial_key(ft_parent))
                if len(parent) > 0:
                    for p in parent:
                        if p.hierarchy == hie:
                            parent = p
                            break
                    if not isinstance(parent, FactorType):
                        issues.append(Issue(itype=IType.ERROR,
                                            description="Parent interface type name '"+ft_parent+"' not found in hierarchy '"+ft_h_name+"'",
                                            location=IssueLocation(sheet_name=name, row=r, column=None)))
                        return
                else:
                    issues.append(Issue(itype=IType.ERROR,
                                        description="Parent interface type name '" + ft_parent + "' not found",
                                        location=IssueLocation(sheet_name=name, row=r, column=None)))
                    return
                # Double check, it must be defined in "hie"
                if ft_parent not in hie.codes:
                    issues.append(Issue(itype=IType.ERROR,
                                        description="Parent interface type name '" + ft_parent + "' not registered in the hierarchy '"+ft_h_name+"'",
                                        location=IssueLocation(sheet_name=name, row=r, column=None)))
                    return
            else:
                parent = None

            # Check if FactorType exists
            ft = glb_idx.get(FactorType.partial_key(ft_name))
            if len(ft) == 0:
                # TODO Compile and CONSIDER attributes (on the FactorType side)
                roegen_type = None
                if ft_roegen_type:
                    roegen_type = FlowFundRoegenType.flow if strcmp(ft_roegen_type, "flow") else FlowFundRoegenType.fund

                ft = FactorType(ft_name,
                                parent=parent, hierarchy=hie,
                                roegen_type=roegen_type,
                                tags=None,  # No tags
                                attributes=dict(unit=ft_unit, description=ft_description, **ft_attributes),
                                expression=ft_formula,
                                sphere=ft_sphere,
                                opposite_processor_type=ft_opposite_processor_type
                                )
                # Simple name
                glb_idx.put(FactorType.partial_key(ft_name, ft.ident), ft)
                if not strcmp(ft_name, ft.full_hierarchy_name()):
                    glb_idx.put(FactorType.partial_key(ft.full_hierarchy_name(), ft.ident), ft)
            else:
                issues.append(Issue(itype=IType.ERROR,
                                    description="Interface type name '" + ft_name + "' already registered",
                                    location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                return

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