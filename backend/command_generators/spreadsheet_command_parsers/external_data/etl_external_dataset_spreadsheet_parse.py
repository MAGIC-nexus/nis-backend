from backend.common.helper import obtain_dataset_source, obtain_dataset_metadata, create_dictionary, strcmp
from backend.domain import get_case_study_registry_objects


def parse_etl_external_dataset_command(sh, area, dataset_name, state):
    """
    Check that the syntax of the input spreadsheet is correct
    Return the analysis in JSON compatible format, for execution

    :param sh:   Input worksheet
    :param area: Area of the input worksheet to be analysed
    :return:     The command in a dict-list object (JSON ready)
    """
    def obtain_column(cn, r1, r2):
        """
        Obtain a list with the values of a column, in the range of rows [r1, r2)

        :param cn: Column number
        :param r1: Starting row
        :param r2: End+1 row
        :return: list with the cell values
        """
        lst = []
        for row in range(r1, r2):
            value = sh.cell(row=row, column=cn).value
            if value is None:
                continue
            lst.append(value)
        return lst

    issues = []
    # Global variables (at parse time they may not be defined, so process carefully...)
    glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
    # Dataset source
    source = obtain_dataset_source(dataset_name)
    # Obtain metadata
    dims, attrs, meas = obtain_dataset_metadata(dataset_name, source)
    # Load all code lists in a temporary dictionary of sets
    # Also check if there is a TIME dimension in the dataset
    cl = create_dictionary()
    we_have_time = False
    for d in dims:
        if dims[d].code_list:
            cl[d] = [k.lower() for k in dims[d].code_list.keys()]  # Attach the code list
        else:
            cl[d] = None  # No code list (TIME_PERIOD for instance)
        if dims[d].istime:
            we_have_time = True
    # Add matching mappings as more dimensions
    for m in mappings:
        if strcmp(mappings[m].source, source) and \
                strcmp(mappings[m].dataset, dataset_name) and \
                mappings[m].origin in dims:
            # Add a dictionary entry for the new dimension, add also the codes present in the map
            cl[mappings[m].destination] = set([t[1] for t in mappings[m].map])

    # Scan columns for Dimensions, Measures and Aggregation.
    # Pivot Table is a Visualization, so now it is not in the command, there will be a command aside.
    # TODO The result COULD be an automatic BI cube (with a separate field)
    # TODO - Write into a set of tables in Mondrian
    # TODO - Generate Schema for Mondrian
    # TODO - Write the Schema for Mondrian
    measures = []
    out_dims = []
    agg_funcs = []
    filter_ = {}  # Cannot use "create_dictionary()" because CaseInsensitiveDict is NOT serializable (which is a requirement)
    result_name = None  # By default, no name for the result. It will be dynamically obtained
    for c in range(area[2], area[3]):
        col_name = sh.cell(row=1, column=c).value
        if not col_name:
            continue

        if col_name.lower().strip() in ["dimensions_kept", "dims", "dimensions"]:  # "GROUP BY"
            lst = obtain_column(c, area[0] + 1, area[1])
            for d in lst:
                if d not in cl:
                    issues.append((3, "The dimension specified for output, '"+d+"' is neither a dataset dimension nor a mapped dimension. ["+', '.join([d2 for d2 in cl])+"]"))
                else:
                    out_dims.append(d)
        elif col_name.lower().strip() in ["aggregation_function", "aggfunc", "agg_func"]:  # "SELECT AGGREGATORS"
            lst = obtain_column(c, area[0] + 1, area[1])
            for f in lst:
                if f.lower() not in ["sum", "average", "avg"]:
                    issues.append((3, "The specified aggregation function, '"+f+"' is not one of the supported ones: 'sum' or 'average'"))
                else:
                    agg_funcs.append(f)
        elif col_name.lower().strip() in ["measures"]:  # "SELECT"
            lst = obtain_column(c, area[0] + 1, area[1])
            # Check for measures
            # TODO (and attributes?)
            for m in lst:
                if m not in meas:
                    issues.append((3, "The specified measure, '"+m+"' is not a measure available in the dataset. ["+', '.join([m2 for m2 in measures])+"]"))
                else:
                    measures.append(m)
        elif col_name in cl:  # A dimension -> "WHERE"
            # Check codes, and add them to the "filter"
            lst = obtain_column(c, area[0] + 1, area[1])
            for cd in lst:
                if cd.lower() not in cl[col_name]:
                    issues.append((3, "The code '"+cd+"' is not present in the codes declared for dimension '"+col_name+"'. Please, check them."))
                else:
                    if col_name not in filter_:
                        lst2 = []
                        filter_[col_name] = lst2
                    else:
                        lst2 = filter_[col_name]
                    lst2.append(cd)
        elif we_have_time and col_name.lower() in ["startperiod", "endperiod"]:  # SPECIAL "WHERE" FOR TIME
            # TODO Instead, should use a single column, "Time", using the interval syntax of the Time column in the Data Input command
            # Interval of time periods
            lst = obtain_column(c, area[0] + 1, area[1])
            if len(lst) > 0:
                filter_[col_name] = lst[0]  # In this case it is not a list, but a number or string !!!!
        elif col_name.lower() in ["result_name", "result name", "resultname"]:
            lst = obtain_column(c, area[0] + 1, area[1])
            if len(lst) > 0:
                result_name = lst[0]

    if len(measures) == 0:
        issues.append((3, "At least one measure should be specified"))

    if len(agg_funcs) == 0:
        issues.append((2, "No aggregation function specified. Assuming 'average'"))
        agg_funcs.append("average")

    if not result_name:
        result_name = source + "_" + dataset_name
        issues.append((2, "No result name specified. Assuming '"+result_name+"'"))

    content = {"dataset_source": source,
               "dataset_name": dataset_name,
               "where": filter_,
               "dimensions": [d for d in dims],
               "group_by": out_dims,
               "measures": measures,
               "agg_funcs": agg_funcs,
               "result_name": result_name
               }
    return issues, None, content

