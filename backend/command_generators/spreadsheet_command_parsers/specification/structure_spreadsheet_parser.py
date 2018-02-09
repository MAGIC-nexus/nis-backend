"""
Declaration of Observables and relations between Processors and/or Factors

"""
import collections

from backend.command_generators import basic_elements_parser
from backend.common.helper import strcmp, create_dictionary


def parse_structure_command(sh, area):
    """
    Analyze the input to produce a JSON object with a list of Observables and relations to other Observables

    Result:[
            {"origin": <processor or factor>,
             "description": <label describing origin>,
             "attributes": {"<attr>": "value"},
             "default_relation": <default relation type>,
             "dests": [
                {"name": <processor or factor>,
                 Optional("relation": <relation type>,)
                 "weight": <expression resulting in a numeric value>
                }
             }
            ]
    :param sh: Input worksheet
    :param area: Tuple (top, bottom, left, right) representing the rectangular area of the input worksheet where the
    command is present
    :return: list of issues (issue_type, message), command label, command content
    """
    some_error = False
    issues = []

    # Scan the sheet, the first column must be one of the keys of "k_list", following
    # columns can contain repeating values
    col_names = {("origin",): "origin",
                 ("relation",): "default_relation",
                 ("destination",): "destination",
                 ("origin label", "label"): "description"
                 }
    # Check columns
    col_map = collections.OrderedDict()
    for c in range(area[2], area[3]):
        col_name = sh.cell(row=area[0], column=c).value
        for k in col_names:
            if col_name.lower() in k:
                col_map[c] = col_names[k]
                break

    # Map key to a list of values
    content = []  # Dictionary of lists, one per metadata key
    for r in range(area[0], area[1]):
        item = {}
        for c in col_map:
            value = sh.cell(row=r, column=c).value
            if not value:
                continue

            k = col_map[c]
            if k == "origin":  # Mandatory
                # Check syntax
                try:
                    basic_elements_parser.string_to_ast(basic_elements_parser.simple_ident, value)
                    item[k] = value
                except:
                    some_error = True
                    issues.append((3, "The name specified for the origin element, '" + value + "', is not valid, in row " + str(r) + ". It must be either a processor or a factor name."))
            elif k == "relation":  # Optional (if not specified, all destinations must specify it)
                # Check syntax
                if value in ('|', '>', '<', '|>', '|<', '<|', '>|', '||>', '||<', '>||', '<||'):
                    if value == '<|': value = '|<'
                    elif value == '>|': value = '|>'
                    elif value == '>||': value = '||>'
                    elif value == '<||': value = '||<'
                    item[k] = value
                else:
                    some_error = True
                    issues.append((3, "The Default relation type specified for the origin element, '" + value + "', is not valid, in row " + str(r) + ". It must be one of '|', '>', '<', '|>', '|<', '||>', '||<'."))
            if k == "destination":  # Mandatory
                # TODO Check syntax. It can contain: a weight, a relation type, a processor or factor name.
                # TODO name. w name. relation name. w relation name
                try:
                    basic_elements_parser.string_to_ast(basic_elements_parser.simple_ident, value)
                    dest = None
                    relation = None
                    weight = None

                    if k not in item:
                        lst = []
                        item[k] = lst
                    else:
                        lst = item[k]
                    if relation == '<|': relation = '|<'
                    elif relation == '>|': relation = '|>'
                    elif relation == '>||': relation = '||>'
                    elif relation == '<||': relation = '||<'
                    lst.append(dict(name=dest, relation=relation, weight=weight))
                    item[k] = value
                except:
                    some_error = True
                    issues.append((3, "The specification of destination, '" + value + "', is not valid, in row " + str(r) + ". It is a sequence of weight (optional) relation (optional) destination element (mandatory)"))
            elif k == "description":  # Optional
                item[k] = value

        # Check parameter completeness before adding it to the list of parameters
        if "origin" not in item:
            issues.append((3, "The element must have a Name, row "+str(r)))
            continue

        content.append(item)

    return issues, None, content


