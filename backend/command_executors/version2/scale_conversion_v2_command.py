import json

from backend.command_generators import Issue, IssueLocation
from backend.common.helper import strcmp
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts import Observer, FactorTypesRelationUnidirectionalLinearTransformObservation, \
    FactorType


class ScaleConversionV2Command(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        def process_line(item):
            sc_src_hierarchy = item.get("source_hierarchy", None)
            sc_src_interface_type = item.get("source_interface_type", None)
            sc_tgt_hierarchy = item.get("target_hierarchy", None)
            sc_tgt_interface_type = item.get("target_interface_type", None)
            sc_scale = item.get("source_hierarchy", None)
            sc_src_context = item.get("source_context", None)
            sc_tgt_context = item.get("target_context", None)
            sc_src_unit = item.get("source_unit", None)
            sc_tgt_unit = item.get("target_unit", None)

            # Check the existence of the interface types

            force_create = True
            if force_create:
                pass

            # Check if FactorTypes exist
            fts = []
            for i, t in enumerate([(sc_src_hierarchy, sc_src_interface_type),
                                   (sc_tgt_hierarchy, sc_tgt_interface_type)]):
                m = "origin" if i == 0 else "destination"
                if not t[1]:
                    issues.append(Issue(itype=3,
                                        description="The "+m+ "interface type name has not been specified",
                                        location=IssueLocation(sheet_name=name, row=r, column=None)))
                    return

                # Check if FactorType exists
                ft = glb_idx.get(FactorType.partial_key(t[1]))
                if len(ft) > 0:
                    if len(ft) == 1:
                        fts.append(ft[0])
                    else:
                        if not t[0]:
                            issues.append(Issue(itype=3,
                                                description="The hierarchy of the " + m + "interface type name has not been specified and the interface type name is not unique",
                                                location=IssueLocation(sheet_name=name, row=r, column=None)))
                            return

                        for ft2 in ft:
                            if strcmp(ft2.hierarchy.name, t[0]):
                                fts.append(ft2)

            if len(fts) != 2:
                issues.append(Issue(itype=3,
                                    description="Found "+str(len(fts))+" interface types in the specification of a scale change",
                                    location=IssueLocation(sheet_name=name, row=r, column=None)))
                return

            # Check that the interface types are from different hierarchies (warn if not; not error)
            if fts[0].hierarchy == fts[1].hierarchy:
                issues.append(Issue(itype=2,
                                    description="The interface types '"+fts[0].name+"' and '"+fts[1].name+"' are in the same hierarchy",
                                    location=IssueLocation(sheet_name=name, row=r, column=None)))

            # Create the directed Scale (Linear "Transformation") Relationship
            origin = fts[0]
            destination = fts[1]
            FactorTypesRelationUnidirectionalLinearTransformObservation.\
                create_and_append(origin, destination, sc_scale,
                                  sc_src_context, sc_tgt_context,
                                  Observer.no_observer_specified)

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