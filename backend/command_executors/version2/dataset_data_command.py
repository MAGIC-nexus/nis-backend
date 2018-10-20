import json
from io import StringIO
import pandas as pd

from backend.command_generators import Issue
from backend.command_generators.spreadsheet_command_parsers_v2 import IssueLocation
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.common.helper import strcmp, prepare_dataframe_after_external_read
from backend.models.statistical_datasets import Dataset


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

        # Process parsed information
        for r, line in enumerate(self._content["items"]):
            # A dataset
            dataset_name = line["name"]
            # Find it in the already available datasets. MUST EXIST
            for n in ds_names:
                if strcmp(dataset_name, n):
                    df = pd.read_json(StringIO(line["values"]), orient="split")
                    # Check columns
                    ds = datasets[n]
                    iss = prepare_dataframe_after_external_read(ds, df)
                    for issue in iss:
                        issues.append(
                            Issue(itype=3,
                                  description=issue,
                                  location=IssueLocation(sheet_name=name, row=-1, column=-1)))
                    # Everything ok? Store the dataframe!
                    if len(iss) == 0:
                        ds.data = df
                    break
            else:
                issues.append(
                    Issue(itype=3,
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