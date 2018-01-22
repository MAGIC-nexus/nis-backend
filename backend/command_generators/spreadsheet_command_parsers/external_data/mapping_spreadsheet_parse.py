from backend.common.helper import strcmp
from backend.command_executors.external_data.mapping_command import MappingCommand
from backend.command_generators import basic_elements_parser


def parse_mapping_command(sh, area, origin, destination):
    """
    Map from a set of categories from an external dataset into a set of MuSIASEM categories
    If the categories do not exist, they are created flat. Later they can be turned into a hierarchy and the mapping
    will still hold

    The mapping has to be MANY to ONE
    The mapping has to be complete (all elements from left side must be covered)


    :param sh: Input worksheet
    :param area: Tuple (top, bottom, left, right) representing the rectangular area of the input worksheet where the
    command is present
    :param origin:
    :param destination:
    :return: list of issues (issue_type, message), command label, command content
    """
    some_error = False
    issues = []
    # Origin
    cell = sh.cell(row=area[0], column=area[2])
    col_name = cell.value
    if origin:
        if not strcmp(origin, col_name):
            some_error = True
            issues.append((3, "The Origin name is different in the sheet name and in the worksheet ("+origin+", "+col_name+")"))
    else:
        origin = col_name
    # Destination
    cell = sh.cell(row=area[0], column=area[2] + 1)
    col_name = cell.value
    if destination:
        if not strcmp(destination, col_name):
            some_error = True
            issues.append((3, "The Destination name is different in the sheet name and in the worksheet (" + destination + ", " + col_name + ")"))
    else:
        destination = col_name

    # Check that origin and destination heterarchy names are syntactically valid: [namespace::]simple_id(.simple_id)*
    for d, c in [("Origin", origin), ("Destination", destination)]:
        try:
            basic_elements_parser.h_name.parseString(c, parseAll=True)
        except:
            some_error = True
            issues.append((3, d + " '" + c + "' has to be a composed identifier"))

    if len(issues) > 0:  # Issues at this point are errors, return if there are any
        return None, issues

    o = []
    d = []
    exp = []
    for r in range(area[0] + 1, area[1]):
        o_value = sh.cell(row=r, column=area[2]).value
        d_value = sh.cell(row=r, column=area[2] + 1).value
        try:
            exp_value = sh.cell(row=r, column=area[2] + 2).value
        except:
            exp_value = None
        if not o_value or not d_value:
            if (not o_value and d_value) or (o_value and not d_value):
                issues.append((2, "Row "+str(r)+": either Origin or Destination is not defined. Row skipped."))
            continue
        o.append(o_value)
        d.append(d_value)
        exp.append(exp_value)
    the_map = [{"o": k, "d": v, "e": e} for k, v, e in zip(o, d, exp)]
    content = {"origin": origin,  # Name of the origin heterarchy
               "destination": destination,  # Name of the destination heterarchy
               "map": the_map  # List of dictionaries
               }
    label = content["origin"] + " - " + content["destination"]
    if True:
        return issues, label, content
    else:
        if not some_error:
            cmd = MappingCommand(label)
            cmd.json_deserialize(content)
        else:
            cmd = None
        return cmd, issues
