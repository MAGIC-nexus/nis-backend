import json
from collections import OrderedDict

import pandas as pd
import numpy as np

import backend
from backend.model_services import IExecutableCommand, get_case_study_registry_objects
from backend.common.helper import obtain_dataset_metadata, strcmp, create_dictionary, \
    augment_dataframe_with_mapped_columns


def obtain_reverse_codes(mapped, dst):
    """
    Given the list of desired dst codes and an extensive map src -> dst,
    obtain the list of src codes

    :param mapped: Correspondence between src codes and dst codes [{"o", "to": [{"d", "e"}]}]
    :param dst: Iterable of destination codes
    :return: List of origin codes
    """
    src = set()
    dest_set = set([d.lower() for d in dst])  # Destination categories
    # Obtain origin categories referencing "dest_set" destination categories
    for k in mapped:
        for t in k["to"]:
            if t["d"] and t["d"].lower() in dest_set:
                src.add(k["o"])
    return list(src)  # list(set([k[0].lower() for k in mapped if k[1].lower() in dest_set]))


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
                for m in mappings:
                    if strcmp(mappings[m].destination, dim) and \
                            strcmp(mappings[m].source, source) and \
                            strcmp(mappings[m].dataset, dataset_name) and \
                            mappings[m].origin in dims:
                        native_dim = mappings[m].origin
                        lst = obtain_reverse_codes(mappings[m].map, lst)
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
        # TODO Prepare an "m" containing ALL the mappings affecting "df"
        # TODO df2 = augment_dataframe_with_mapped_columns(df, m, ["value"])
        for m in mappings:
            if strcmp(mappings[m].source, source) and \
                    strcmp(mappings[m].dataset, dataset_name) and \
                    mappings[m].origin in dims:
                # TODO Change by n-to-m mapping
                # Elaborate a many to one mapping
                tmp = []
                for el in mappings[m].map:
                    for to in el["to"]:
                        if to["d"]:
                            tmp.append([el["o"], to["d"]])
                df_dst = pd.DataFrame(tmp, columns=['sou_rce', mappings[m].destination.lower()])
                for di in df.columns:
                    if strcmp(mappings[m].origin, di):
                        d = di
                        break
                # df[d] = df[d].str.upper()  # Upper case column before merging
                df = pd.merge(df, df_dst, how='left', left_on=d, right_on='sou_rce')
                del df['sou_rce']

        # Aggregate (If any dimension has been specified)
        if len(self._content["group_by"]) > 0:
            # Column names where data is
            # HACK: for the case where the measure has been named "obs_value", use "value"
            values = [m.lower() if m.lower() != "obs_value" else "value" for m in self._content["measures"]]
            rows = [v.lower() for v in self._content["group_by"]]  # Group by dimension names
            aggs = []  # Aggregation functions
            for f in self._content["agg_funcs"]:
                if f.lower() in ["avg", "average"]:
                    aggs.append(np.average)
                elif f.lower() in ["sum"]:
                    aggs.append(np.sum)
            # Calculate Pivot Table. The columns are a combination of values x aggregation functions
            # For instance, if two values ["v2", "v2"] and two agg. functions ["avg", "sum"] are provided
            # The columns will be: [["average", "v2"], ["average", "v2"], ["sum", "v2"], ["sum", "v2"]]
            try:
                # Check that all "rows" on which pivot table aggregates are present in the input "df"
                # If not either synthesize them (only if there is a single filter value) or remove (if not present
                for r in rows.copy():
                    if r not in df.columns:
                        found = False
                        for k in params:
                            if k.lower() == r:
                                found = True
                                if len(params[k]) == 1:
                                    df[r] = params[k][0]
                                else:
                                    rows.remove(r)
                                    issues((2, "Dimension '" + r + "' removed from the list of dimensions because it is not present in the raw input dataset."))
                                break
                        if not found:
                            rows.remove(r)
                            issues((2, "Dimension '" + r + "' removed from the list of dimensions because it is not present in the raw input dataset."))
                # Pivot table using Group by
                if True:
                    groups = df.groupby(by=[rows], as_index=False)  # Split
                    lst = []
                    for v in values:
                        if agg
                        lst.append((v, ))
                    groups.agg()
                    # agg({"d": ['size', 'count', np.nansum]})
                    # size(), sum(),
                else:
                    # Pivot table
                    df2 = pd.pivot_table(df,
                                         values=values,
                                         index=rows,
                                         aggfunc=[aggs[0]], fill_value=np.NaN, margins=False,
                                         dropna=True)
                # Remove the multiindex in columns
                df2.columns = [col[-1] for col in df2.columns.values]
                # Remove the index
                df2.reset_index(inplace=True)
                # The result, all columns (no index), is stored for later use
                ds.data = df2
            except Exception as e:
                issues.append((3, "There was a problem: "+str(e)))

        # Store the dataset in State
        datasets[result_name] = ds

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

