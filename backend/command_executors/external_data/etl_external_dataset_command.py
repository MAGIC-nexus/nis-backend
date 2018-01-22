import json

import pandas as pd
import numpy as np

import backend
from backend.domain import IExecutableCommand, get_case_study_registry_objects
from backend.common.helper import obtain_dataset_metadata, strcmp, create_dictionary


def obtain_reverse_codes(mapped, dst):
    """
    Given the list of desired dst codes and an extensive map src -> dst,
    obtain the list of src codes

    :param mapped: Correspondence between src codes and dst codes
    :param dst: List of destination codes
    :return: List of origin codes
    """
    src = set()
    lst = [d.upper() for d in dst]
    for k in mapped:
        if k[1] in lst:
            src.add(k[0])

    return list(src)


class ETLExternalDatasetCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        """
        First bring the data considering the filter
        Second, group, third aggregate
        Finally, store the result in State
        """
        issues = []
        # Obtain global variables in state
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)

        # DS Source + DS Name
        source = self._content["dataset_source"]
        dataset_name = self._content["dataset_name"]
        # Result name
        result_name = self._content["result_name"]
        if result_name in datasets or state.get(result_name):
            issues.append((2, "A dataset called '"+result_name+"' is already stored in the registry of datasets"))
        # Dataset metadata
        dims, attrs, meas = obtain_dataset_metadata(dataset_name, source)
        # Obtain filter parameters
        params = create_dictionary()  # Native dimension name to list of values the filter will allow to pass
        for dim in self._content["where"]:
            lst = self._content["where"][dim]
            native_dim = None
            if dim.lower() in ["startperiod", "endperiod"]:
                native_dim = dim
                lst = [lst]
            elif dim not in dims:
                # Check if there is a mapping. If so, obtain the native equivalent(s). If not, ERROR
                mapping = None
                for m in mappings:
                    if strcmp(mappings[m].destination, dim) and \
                            strcmp(mappings[m].source, source) and \
                            strcmp(mappings[m].dataset_name, dataset_name) and \
                            mappings[m].origin in dims:
                        native_dim = mappings[m].origin
                        lst = obtain_reverse_codes(mapping[m].map, lst)
                        break
            else:
                native_dim = dim
            if native_dim:
                if native_dim not in params:
                    f = set()
                    params[native_dim] = f
                else:
                    f = params[native_dim]
                f.update(lst)
        # Convert param contents from set to list
        for p in params:
            params[p] = [i for i in params[p]]

        # Obtain the filtered Dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ds = backend.data_source_manager.get_dataset_filtered(source, dataset_name, params)
        df = ds.data

        # Join with mapped dimensions (augment it)
        for m in mappings:
            if strcmp(mappings[m].source, source) and \
                    strcmp(mappings[m].dataset_name, dataset_name) and \
                    mappings[m].origin in dims:
                df_dst = pd.DataFrame(mappings[m].maps, columns=['sou_rce', mappings[m].destination])
                for di in df.columns:
                    if strcmp(mappings[m].origin, di):
                        d = di
                        break
                # df[d] = df[d].str.upper()  # Upper case column before merging
                df = pd.merge(df, df_dst, how='left', left_on=d, right_on='sou_rce')
                del df['sou_rce']
                break

        # Aggregate (If any dimension has been specified)
        if len(self._content["group_by"]) > 0:
            values = ["value"] # TODO self._content["measures"]  # Column names where data is
            rows = [v.lower() for v in self._content["group_by"]]  # Group by dimension names
            aggs = []  # Aggregation functions
            for f in self._content["agg_funcs"]:
                if f in ["avg", "average"]:
                    aggs.append(np.average)
                elif f in ["sum"]:
                    aggs.append(np.sum)
            # Calculate Pivot Table. The columns are a combination of values x aggregation functions
            # For instance, if two values ["v1", "v2"] and two agg. functions ["avg", "sum"] are provided
            # The columns will be: [["average", "v1"], ["average", "v2"], ["sum", "v1"], ["sum", "v2"]]
            df2 = pd.pivot_table(df,
                                 values=values,
                                 index=rows,
                                 aggfunc=aggs, fill_value=np.NaN, margins=False,
                                 dropna=True)
            ds.data = df2

        # Store the dataset in State
        datasets[dataset_name] = ds
        state.set(dataset_name, ds)

        return issues, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the metadata dictionary
        return self._content

    def json_deserialize(self, json_input):
        # TODO Read and check keys validity
        issues = []
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)

        return issues


