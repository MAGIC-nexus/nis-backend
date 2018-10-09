import json

from backend.command_generators import Issue
from backend.command_generators.parser_ast_evaluators import dictionary_from_key_value_list
from backend.command_generators.spreadsheet_command_parsers_v2 import IssueLocation
from backend.common.helper import create_dictionary
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.models.musiasem_concepts_helper import convert_hierarchy_to_code_list
from backend.models.statistical_datasets import Dataset, Dimension


class DatasetDefCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        def process_line(item):
            # Read variables
            dsd_dataset_name = item.get("dataset_name", None)
            dsd_dataset_data_location = item.get("dataset_data_location", None)
            dsd_concept_type = item.get("concept_type", None)
            dsd_concept_name = item.get("concept_name", None)
            dsd_concept_data_type = item.get("concept_data_type", None)
            dsd_concept_domain = item.get("concept_domain", None)
            dsd_concept_description = item.get("concept_description", None)
            dsd_attributes = item.get("concept_attributes", None)
            if dsd_attributes:
                try:
                    attributes = dictionary_from_key_value_list(dsd_attributes, glb_idx)
                except Exception as e:
                    issues.append(Issue(itype=3,
                                        description=str(e),
                                        location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                    return
            else:
                attributes = {}

            if dsd_dataset_name in ds_names:
                issues.append(Issue(itype=3,
                                    description="The dataset '"+dsd_dataset_name+"' has been already defined",
                                    location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                return

            # Internal dataset definitions cache
            ds = current_ds.get(dsd_dataset_name, None)
            if True:  # Statistical dataset format
                if not ds:
                    ds = Dataset()
                    ds.code = dsd_dataset_name  # Name
                    if not dsd_concept_type:
                        attributes["_location"] = dsd_dataset_data_location  # Location
                        ds.description = dsd_concept_description
                        ds.attributes = attributes  # Set attributes
                    ds.database = None
                    current_ds[dsd_dataset_name] = ds
                # If concept_type is defined => add a concept
                if dsd_concept_type:
                    d = Dimension()
                    d.dataset = ds
                    d.description = dsd_concept_description
                    d.code = dsd_concept_name
                    d.is_measure = False if dsd_concept_type.lower() == "dimension" else True
                    if not d.is_measure and dsd_concept_data_type.lower() == "time":
                        d.is_time = True
                    else:
                        d.is_time = False
                    if dsd_concept_type.lower() == "attribute":
                        attributes["_attribute"] = True
                    else:
                        attributes["_attribute"] = False
                    if dsd_concept_data_type.lower() == "category":
                        h = hierarchies.get(dsd_concept_domain, None)
                        if not h:
                            issues.append(Issue(itype=3,
                                                description="Could not find hierarchy of Categories '" + dsd_concept_domain + "'",
                                                location=IssueLocation(sheet_name=name, row=r + 1, column=None)))
                            return
                        d.hierarchy = h
                        # Reencode the Hierarchy as a CodeList
                        cl = convert_hierarchy_to_code_list(h)
                        d.code_list = cl
                    attributes["_datatype"] = dsd_concept_data_type
                    attributes["_domain"] = dsd_concept_domain
                    d.attributes = attributes

        issues = []
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        name = self._content["command_name"]

        # List of available dataset names. The newly defined datasets must not be in this list
        ds_names = [ds.name for ds in datasets]

        # List of available Category hierarchies
        hierarchies = create_dictionary()
        for h in hh:
            hierarchies[h.name] = hh

        # Datasets being defined in this Worksheet
        current_ds = create_dictionary()

        # Process parsed information
        for r, line in enumerate(self._content["items"]):
            # If the line contains a reference to a dataset or hierarchy, expand it
            # If not, process it directly
            is_expansion = False
            if is_expansion:
                pass
            else:
                process_line(line)

        # Any error?
        for issue in issues:
            if issue.itype == 3:
                error = True
                break
        else:
            error = False

        if not error:
            # If no error happened, add the new Datasets to the Datasets in the "global" state
            for ds in current_ds:
                datasets[ds] = current_ds[ds]

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