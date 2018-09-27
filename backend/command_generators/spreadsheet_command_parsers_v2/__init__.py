from attr import attrs, attrib
from typing import List, Tuple
from backend import CommandField
from backend.command_field_definitions import commands
from backend.command_generators import Issue, basic_elements_parser


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

    content = []  # The output JSON
    # Each row
    for r in range(area[0] + 1, area[1]):
        line = {}

        # Mandatory values
        mandatory_not_found = set([c.name for c in cols if c.mandatory])

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
                        basic_elements_parser.string_to_ast(col.parser, value)
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

        if len(mandatory_not_found) > 0:
            issues.append(Issue(itype=3,
                                description="Mandatory columns: " + ", ".join(
                                    mandatory_not_found) + " have not been specified",
                                location=IssueLocation(sheet_name=name, row=r, column=None)))
        else:
            content.append(line)

    return issues, None, {"items": content, "command_name": name}
