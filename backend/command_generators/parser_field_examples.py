from backend.command_generators.parser_field_parsers import simple_h_name, simple_ident, alphanums_string, code_string, \
    unquoted_string, time_expression, reference, key_value_list, arith_boolean_expression, \
    value, expression_with_parameters, list_simple_ident, domain_definition, unit_name, key_value, processor_names, \
    url_parser

generic_field_examples = {
    simple_ident: ["p1", "wind_farm", "WF1"],
    simple_h_name: ["p1", "p1.p2", "wind_farm", "WF1.WindTurbine"],
    alphanums_string: ["abc", "123", "abc123", "123abc"],
    code_string: ["a1", "a-1", "10-011", "a_23"],
    unquoted_string: ["<any string>"],
    # hierarchy_expression_v2: ["3*(5-2)", ],
    time_expression: ["Year", "2017", "Month", "2017-05 - 2018-01", "5-2017", "5-2017 - 2018-1", "2017-2018"],
    reference: ["[Auth2017]", "Auth2017", "Roeg1971"],
    key_value_list: ["level='n+1'", "a=5*p1, b=[Auth2017], c=?p1>3 -> 'T1', p1<=3 -> 'T2'?"],
    key_value: ["level='n+1'", "a=5*p1", "b=[Auth2017]", "c=?p1>3 -> 'T1', p1<=3 -> 'T2'?"],
    value: ["p1*3.14", "5*(0.3*p1)", "?p1>3 -> 'T1', p1<=3 -> 'T2'?", "a_code"],
    expression_with_parameters: ["p1*3.14", "5*(0.3*p1)", "?p1>3 -> 'T1', p1<=3 -> 'T2'?", "a_code"],
    list_simple_ident: ["p1", "p1, wind_farm"],
    domain_definition: ["p1", "wind_farm", "[-3.2, 4)", "(1, 2.3]"],
    unit_name: ["kg", "m^2", "ha", "hours", "km^2"],
    processor_names: ["{a}b", "{a}b{c}", "aa{b}aa", "{a}", "..", "..Crop", "Farm..Crop", "Farm..", "..Farm.."],
    url_parser: ["https://www.magic-nexus.eu", "https://nextcloud.data.magic-nexus.eu/", "https://jupyter.data.magic-nexus.eu"]
}

generic_field_syntax = {
    simple_ident: "Simple name. A sequence of letters, numbers and/or '_', starting with a letter",
    simple_h_name: "Hierarchical name. One or more 'simple names' (sequence of letters, numbers and/or '_' starting "
                   "with a letter), separated by '.'",
    alphanums_string: "Alphanumeric string. A sequence of letters and/or numbers",
    code_string: "Code string. A sequence of letters, numbers and/or '_' '-', starting with a letter",
    unquoted_string: "Unquoted string. Any string",
    # hierarchy_expression_v2: ["3*(5-2)", ],
    time_expression: "Time expression. Serves to specify a year or a month, or an interval from year to year, or from "
                     "month to month when the value is the accumulation for the period. It allows the generic 'Year' "
                     "or 'Month' when the information is valid",
    reference: "Reference. An optionally square bracketed '[]' simple name (sequence of letters, numbers and/or '_' "
               "starting with a letter)",
    key_value_list: "Key=Value list. A list of comma separated 'Key=Value', where 'Key' is a simple name and 'Value' "
                    "an arithmetic-boolean expression optionally referencing previously declared parameters",
    key_value: "Key-Value. 'Key' is a simple name and 'Value' an arithmetic-boolean expression optionally referencing "
               "previously declared parameters",
    value: "Value. An arithmetic-boolean expression optionally referencing previously declared parameters",
    expression_with_parameters: "Arithmetic-boolean expression. A formula combining parameters, numbers, strings and "
                                "booleans with arithmetic (+-*/) and boolean (AND, OR, NOT) operators. Some functions "
                                "are also implemented (sin, cos currently)",
    list_simple_ident: "List of simple names. A comma separated list of simple names (sequence of letters, numbers "
                       "and/or '_' starting with a letter)",
    domain_definition: "Domain definition. Either the name of a Code Hierarchy or a numeric interval, using [] for "
                       "closed and () for open",
    unit_name: "Unit name. The abbreviation of a valid unit name.",
    processor_names: "Processors. An expression allowing the specification of several processors in a row. ",
    url_parser: "URL. A valid URL (search for URL syntax elsewhere)"
}
