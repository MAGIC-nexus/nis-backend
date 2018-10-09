from attr import attrs, attrib
from typing import List, Tuple
from backend import CommandField
from backend.command_field_definitions import commands
from backend.command_generators import Issue, parser_field_parsers
from backend.common.helper import strcmp


@attrs
class IssueLocation:
    sheet_name = attrib()
    row = attrib()
    column = attrib()


def check_columns(sh, name: str, area: Tuple, cols: List[CommandField], command_name: str, ignore_not_found=False):
    """
    When parsing of a command starts, check columns
    Try to match each column with declared column fields. If a column is not declared, raise an error (or ignore it)
    If mandatory columns are not found, raise an error
    :param sh: The worksheet being analyzed
    :param name: The name of the worksheet
    :param area: Area inside the worksheet that will be scanned
    :param cols: List of CommandField
    :param command_name: A string with the name of the command
    :param ignore_not_found: True if a column not matching declared ones has to be ignored, False if an error has to be raised in this case
    :return: The map column name to column index (or indices for multiply declared columns); The issues found
    """

    issues: List[Issue] = []

    # Set of mandatory columns
    mandatory_not_found = set([c.name for c in cols if c.mandatory])

    # Check columns
    col_map = {}
    for c in range(area[2], area[3]):
        col_name = sh.cell(row=area[0], column=c).value.strip()
        for col in cols:
            if col.regex_allowed_names.match(col_name):
                # Column Name to Column Index
                if not col.many_appearances:
                    col_map[col.name] = c
                else:
                    if col.name not in col_map:
                        col_map[col.name] = []
                    col_map[col.name].append(c)
                # Mandatory found (good)
                if col.name in mandatory_not_found:
                    mandatory_not_found.discard(col.name)
                break
        else:  # No match for the column "col_name"
            if not ignore_not_found:
                issues.append(Issue(itype=3,
                                    description="The column name '" + col_name + "' does not match any of the allowed column names for the command '" + command_name + "'",
                                    location=IssueLocation(sheet_name=name, row=1, column=c)))

    if len(mandatory_not_found) > 0:
        issues.append(Issue(itype=3,
                            description="Mandatory columns: " + ", ".join(
                                mandatory_not_found) + " have not been specified",
                            location=IssueLocation(sheet_name=name, row=1, column=None)))

    return col_map, issues


def parse_command(sh, area, name: str, cmd_name):
    """
    Parse command in general
    Generate a JSON
    Generate a list of issues

    :param sh: Worksheet to read
    :param area: Area of the worksheet
    :param name: Name of the worksheet
    :param cmd_name: Name of the command. Key to access "commands" variable. Also, shown in issue descriptions
    :return: issues List, None, content (JSON)
    """

    issues: List[Issue] = []

    cols = commands[cmd_name]  # List of CommandField that will guide the parsing
    col_map, local_issues = check_columns(sh, name, area, cols, cmd_name)

    if any([i.itype == 3 for i in local_issues]):
        return local_issues, None, None

    issues.extend(local_issues)

    # "mandatory" can be defined as expression depending on other base fields (like in RefBibliographic command fields)
    # Elaborate a list of fields having this "complex" mandatory property
    complex_mandatory_cols = [c for c in cols if isinstance(c.mandatory, str)]

    content = []  # The output JSON
    # Each row
    for r in range(area[0] + 1, area[1]):
        line = {}
        expandable = False  # The line contains at least one field implying expansion into multiple lines
        complex = False  # The line contains at least one field with a complex rule (which cannot be evaluated with a simple cast)

        # Constant mandatory values
        mandatory_not_found = set([c.name for c in cols if c.mandatory and isinstance(c.mandatory, bool)])

        # Each "field"
        for cname in col_map.keys():
            col = next(c for c in cols if c.name == cname)
            value = sh.cell(row=r, column=col_map[cname]).value
            if value:
                if not isinstance(value, str):
                    value = str(value)
                value = value.strip()
            else:
                continue

            if col.allowed_values:
                if value.lower() not in [v.lower() for v in col.allowed_values]:  # TODO Case insensitive CI
                    col_header = sh.cell(row=1, column=col_map[cname]).value
                    issues.append(Issue(itype=3,
                                        description="Field '" + col_header + "' of command '" + cmd_name + "' has as allowed values: "+", ".join(col.allowed_values)+". Entered: " + value,
                                        location=IssueLocation(sheet_name=name, row=r, column=col_map[cname])))
                else:
                    line[cname] = value
            else:
                # Parse. Just check syntax
                if col.parser:
                    try:
                        ast = parser_field_parsers.string_to_ast(col.parser, value)
                        # Rules are in charge of informing if the result is expandable and if it complex
                        if "expandable" in ast and ast["expandable"]:
                            expandable = True
                        if "complex" in ast and ast["complex"]:
                            complex = True

                        line[cname] = value
                    except:
                        col_header = sh.cell(row=1, column=col_map[cname]).value
                        issues.append(Issue(itype=3,
                                            description="The value in field '" + col_header + "' of command '" + cmd_name + "' is not syntactically correct. Entered: " + value,
                                            location=IssueLocation(sheet_name=name, row=r, column=col_map[cname])))
                else:
                    line[cname] = value

            if col.name in mandatory_not_found:
                mandatory_not_found.discard(col.name)

        # Flags to accelerate the second evaluation, during execution
        line["_expandable"] = expandable
        line["_complex"] = complex

        # Append if all mandatory fields have been filled
        may_append = True
        if len(mandatory_not_found) > 0:
            issues.append(Issue(itype=3,
                                description="Mandatory columns: " + ", ".join(
                                    mandatory_not_found) + " have not been specified",
                                location=IssueLocation(sheet_name=name, row=r, column=None)))
            may_append = False

        # Check varying mandatory fields (fields depending on the value of other fields)
        for c in complex_mandatory_cols:
            col = next(c2 for c2 in col_map if strcmp(c.name, c2))
            if isinstance(c.mandatory, str):
                # Evaluate
                mandatory = eval(c.mandatory, None, line)
                may_append = (mandatory and col in line) or (not mandatory)
                if mandatory and col not in line:
                    issues.append(Issue(itype=3,
                                        description="Mandatory column: " + col + " has not been specified",
                                        location=IssueLocation(sheet_name=name, row=r, column=None)))

        if may_append:
            content.append(line)

    return issues, None, {"items": content, "command_name": name}
