from backend.command_generators import parser_field_parsers, Issue
from backend.command_generators.parser_field_parsers import simple_ident
from backend.command_generators.spreadsheet_command_parsers_v2 import IssueLocation
from backend.common.helper import obtain_dataset_source, obtain_dataset_metadata, create_dictionary, strcmp
from backend.model_services import get_case_study_registry_objects


# TODO Currently is just a copy of "parse_etl_external_dataset_command" function
# TODO It has two new parameters: "InputDataset" and "AvailableAtDateTime"
# TODO Time dimension can be specified as "Time" or as "StartTime" "EndTime"
# TODO Result parameter column also change a bit
# TODO For a reference of fields, see "DatasetQry" command in "MuSIASEM case study commands" Google Spreadsheet

def parse_dataset_qry_command(sh, area, name, state):
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

    # Look for the name of the input Dataset
    dataset_name = None
    available_at_datetime = None
    for c in range(area[2], area[3]):
        col_name = sh.cell(row=1, column=c).value
        if not col_name:
            continue
        if col_name.lower().strip() in ["inputdataset"]:
            lst = obtain_column(c, area[0]+1, area[1])
            for v in lst:
                if v:
                    dataset_name = v
                    break  # Stop on first definition
        elif col_name.lower().strip() in ["availableatdatetime"]:
            lst = obtain_column(c, area[0]+1, area[1])
            for v in lst:
                if v:
                    available_at_datetime = v
                    break  # Stop on first definition

    # Obtain the source
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
            tmp = [to["d"] for o in mappings[m].map for to in o["to"] if to["d"]]
            cl[mappings[m].destination] = set(tmp)  # [t[1] for t in mappings[m].map]

    # Scan columns for Dimensions, Measures and Aggregation.
    # Pivot Table is a Visualization, so now it is not in the command, there will be a command aside.

    # TODO The result COULD be an automatic BI cube (with a separate field)
    # TODO - Write into a set of tables in Mondrian
    # TODO - Generate Schema for Mondrian
    # TODO - Write the Schema for Mondrian

    measures = []
    out_dims = []
    agg_funcs = []
    measures_as = []
    filter_ = {}  # Cannot use "create_dictionary()" because CaseInsensitiveDict is NOT serializable (which is a requirement)
    result_name = None  # By default, no name for the result. It will be dynamically obtained
    for c in range(area[2], area[3]):
        col_name = sh.cell(row=1, column=c).value
        if not col_name:
            continue
        if col_name.lower().strip() in ["resultdimensions", "dimensions"]:  # "GROUP BY"
            lst = obtain_column(c, area[0] + 1, area[1])
            for d in lst:
                if not d:
                    continue
                if d not in cl:
                    issues.append(Issue(itype=3,
                                        description="The dimension specified for output, '"+d+"' is neither a dataset dimension nor a mapped dimension. ["+', '.join([d2 for d2 in cl])+"]",
                                        location=IssueLocation(sheet_name=name, row=c + 1, column=None)))
                else:
                    out_dims.append(d)
        elif col_name.lower().strip() in ["resultsmeasures", "measures"]:  # "SELECT"
            lst = obtain_column(c, area[0] + 1, area[1])
            # Check for measures
            # TODO (and attributes?)
            for m in lst:
                if not m:
                    continue
                if m not in meas:
                    issues.append(Issue(itype=3,
                                        description="The specified measure, '"+m+"' is not a measure available in the dataset. ["+', '.join([m2 for m2 in measures])+"]",
                                        location=IssueLocation(sheet_name=name, row=c + 1, column=None)))
                else:
                    measures.append(m)
        elif col_name.lower().strip() in ["resultmeasuresaggregation", "resultmeasuresaggregator", "aggregation"]:  # "SELECT AGGREGATORS"
            lst = obtain_column(c, area[0] + 1, area[1])
            for f in lst:
                if f.lower() not in ["sum", "avg", "count", "sumna", "countav", "avgna", "pctna"]:
                    issues.append(Issue(itype=3,
                                        description="The specified aggregation function, '"+f+"' is not one of the supported ones: 'sum', 'avg', 'count', 'sumna', 'avgna', 'countav', 'pctna'",
                                        location=IssueLocation(sheet_name=name, row=c + 1, column=None)))
                else:
                    agg_funcs.append(f)
        elif col_name.lower().strip() in ["resultmeasuresas", "measuresas"]:  # "AS <name>"
            lst = obtain_column(c, area[0] + 1, area[1])
            for m in lst:
                measures_as.append(m)
        elif col_name in cl:  # A dimension -> "WHERE"
            # Check codes, and add them to the "filter"
            lst = obtain_column(c, area[0] + 1, area[1])
            for cd in lst:
                if not cd:
                    continue
                if str(cd).lower() not in cl[col_name]:
                    issues.append(Issue(itype=3,
                                        description="The code '"+cd+"' is not present in the codes declared for dimension '"+col_name+"'. Please, check them.",
                                        location=IssueLocation(sheet_name=name, row=c + 1, column=None)))
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
        elif col_name.lower() in ["outputdatasetname", "outputdataset", "result_name", "result name", "resultname"]:
            lst = obtain_column(c, area[0] + 1, area[1])
            if len(lst) > 0:
                result_name = lst[0]
                try:
                    parser_field_parsers.string_to_ast(simple_ident, result_name)
                except:
                    issues.append(Issue(itype=3,
                                        description="Column '" + col_name + "' has an invalid dataset name '" + result_name + "'",
                                        location=IssueLocation(sheet_name=name, row=c + 1, column=None)))

    if len(measures) == 0:
        issues.append(Issue(itype=3,
                            description="At least one measure should be specified",
                            location=IssueLocation(sheet_name=name, row=c + 1, column=None)))

    if len(agg_funcs) == 0:
        issues.append(Issue(itype=2,
                            description="No aggregation function specified. Assuming 'average'",
                            location=IssueLocation(sheet_name=name, row=c + 1, column=None)))
        agg_funcs.append("average")

    if not result_name:
        result_name = source + "_" + dataset_name
        issues.append(Issue(itype=2,
                            description="No result name specified. Assuming '"+result_name+"'",
                            location=IssueLocation(sheet_name=name, row=c + 1, column=None)))

    content = {"dataset_source": source,
               "dataset_name": dataset_name,
               "dataset_datetime": available_at_datetime,
               "where": filter_,
               "dimensions": [d for d in dims],
               "group_by": out_dims,
               "measures": measures,
               "agg_funcs": agg_funcs,
               "measures_as": measures_as,
               "result_name": result_name
               }
    return issues, None, content

