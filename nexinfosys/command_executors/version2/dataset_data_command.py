import json
from io import StringIO
import pandas as pd

from nexinfosys.command_generators import Issue, IssueLocation, IType
from nexinfosys.model_services import IExecutableCommand, get_case_study_registry_objects
from nexinfosys.common.helper import strcmp, prepare_dataframe_after_external_read, create_dictionary
from nexinfosys.models import CodeImmutable
from nexinfosys.models.statistical_datasets import CodeList


class DatasetDataCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        issues = []

        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        name = self._content["command_name"]

        # List of available dataset names. The newly defined datasets must not be in this list
        ds_names = [ds.code for ds in datasets.values()]

        # List of datasets with local worksheet name
        external_dataset_name = None
        for ds in datasets.values():
            if ds.attributes["_location"].lower().startswith("data://#"):
                worksheet = ds.attributes["_location"][len("data://#"):]
                if not worksheet.lower().startswith("datasetdata "):
                    worksheet = "DatasetData " + worksheet

                if strcmp(worksheet, name):
                    external_dataset_name = ds.code

        # Process parsed information
        for r, line in enumerate(self._content["items"]):
            # A dataset
            dataset_name = line["name"]
            if dataset_name == "":
                if external_dataset_name:
                    dataset_name = external_dataset_name
                else:
                    issues.append(Issue(itype=IType.ERROR,
                                        description="The column name 'DatasetName' was not defined for command 'DatasetData' and there is no 'location' in a DatasetDef command pointing to it",
                                        location=IssueLocation(sheet_name=name, row=1, column=None)))

            # Find it in the already available datasets. MUST EXIST
            for n in ds_names:
                if strcmp(dataset_name, n):
                    df = pd.read_json(StringIO(line["values"]), orient="split")
                    # Check columns
                    ds = datasets[n]
                    iss = prepare_dataframe_after_external_read(ds, df, name)
                    issues.extend(iss)
                    # Everything ok? Store the dataframe!
                    if len(iss) == 0:
                        r = ds.attributes["_dataset_first_row"]
                        # Loop over "ds" concepts.
                        # - "dimension" concepts of type "string" generate a CodeHierarchy
                        # - Check that the DataFrame contains ALL declared concepts. If not, generate issue
                        for c in ds.dimensions:
                            if c.code in df.columns:
                                dsd_concept_data_type = c.attributes["_datatype"]
                                if dsd_concept_data_type.lower() == "string" and not c.is_measure:  # Freely defined dimension
                                    cl = df[c.code].unique().tolist()
                                    c.code_list = CodeList.construct(
                                        c.code, c.code, [""],
                                        codes=[CodeImmutable(c, c, "", []) for c in cl]
                                    )
                            else:
                                issues.append(Issue(itype=IType.ERROR,
                                                    description=f"Concept '{c.code}' not defined for '{ds.code}'",
                                                    location=IssueLocation(sheet_name=name, row=r, column=None)))
                        ds.data = df
                    break
            else:
                issues.append(
                    Issue(itype=IType.ERROR,
                          description="Metadata for the dataset '"+dataset_name+"' must be defined previously",
                          location=IssueLocation(sheet_name=name, row=-1, column=-1)))

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
