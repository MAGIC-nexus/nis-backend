import json

from backend.command_generators import Issue
from backend.command_generators.spreadsheet_command_parsers_v2 import IssueLocation
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts import Reference


class ReferencesV2Command(IExecutableCommand):
    """ It is a mere data container
        Depending on the type, the format can be controlled with a predefined schema
    """
    def __init__(self, name: str, ref_type: str):
        self._name = name
        self._content = None
        self._ref_type = ref_type

    def execute(self, state: "State"):
        """
        Process each of the references, simply storing them as Reference objects
        """
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        name = self._content["command_name"]
        issues = []

        # Receive a list of validated references
        # Store them as objects, which can be referred to later
        for ref in self._content["items"]:
            r = ref["_row"]

            if "ref_id" not in ref:
                issues.append(Issue(itype=3,
                                    description="'ref_id' field not found: "+str(ref),
                                    location=IssueLocation(sheet_name=name, row=r, column=None)))
                continue
            else:
                ref_id = ref["ref_id"]
                ref_type = self._ref_type
                existing = glb_idx.get(Reference.partial_key(ref_id, ref_type))
                if len(existing) == 1:
                    issues.append(Issue(itype=3,
                                        description="Reference '"+ref_id+"' of type '"+ref_type+"' is already defined. Not allowed",
                                        location=IssueLocation(sheet_name=name, row=r, column=None)))
                    continue
                elif len(existing) > 1:  # This condition should not occur...
                    issues.append(Issue(itype=3,
                                        description="The reference '"+ref_id+"' of type '"+ref_type+"' is defined more than one time ("+str(len(existing))+")",
                                        location=IssueLocation(sheet_name=name, row=r, column=None)))
                    continue

                # Create and store the Reference
                reference = Reference(ref_id, ref_type, ref)
                glb_idx.put(reference.key(), reference)

        return issues, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the content
        return self._content

    def json_deserialize(self, json_input):
        # TODO Check validity
        issues = []
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)
        return issues


class ProvenanceReferencesCommand(ReferencesV2Command):
    def __init__(self, name: str):
        ReferencesV2Command.__init__(self, name, "prov")


class BibliographicReferencesCommand(ReferencesV2Command):
    def __init__(self, name: str):
        ReferencesV2Command.__init__(self, name, "bib")


class GeographicReferencesCommand(ReferencesV2Command):
    def __init__(self, name: str):
        ReferencesV2Command.__init__(self, name, "geo")