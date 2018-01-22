from backend import metadata_fields
from backend.common.helper import strcmp, create_dictionary
from backend.command_executors.specification.metadata_command import MetadataCommand


def parse_metadata_command(sh, area):
    """
    Most "parse" methods are mostly syntactic (as opposed to semantic). They do not check existence of names.
    But in this case, the valid field names are fixed beforehand, so they are checked at this time.
    Some of the fields will be controlled also, according to some

    :param sh: Input worksheet
    :param area: Tuple (top, bottom, left, right) representing the rectangular area of the input worksheet where the
    command is present
    :return: list of issues (issue_type, message), command label, command content
    """
    some_error = False
    issues = []
    controlled = create_dictionary()
    mandatory = create_dictionary()
    keys = create_dictionary()
    for t in metadata_fields:
        controlled[t[0]] = t[3]
        mandatory[t[0]] = t[2]
        keys[t[0]] = t[4]

    # Scan the sheet, the first column must be one of the keys of "k_list", following
    # columns can contain repeating values

    # Map key to a list of values
    content = {}  # Dictionary of lists, one per metadata key
    for r in range(area[0], area[1]):
        key = sh.cell(row=r, column=area[2]).value
        if key in keys:
            for c in range(area[2]+1, area[3]):
                value = sh.cell(row=r, column=area[2] + 1).value
                if value and str(value).strip():
                    if controlled[key]:
                        # Control "value" if the field is controllable
                        cl = {"dimensions": ["water", "energy", "food", "land", "climate"],
                              "subject_topic_keywords": None,
                              "geographical_level": ["regional", "region", "country", "europe", "sectoral", "sector"],
                              "geographical_situation": None,  # TODO Read the list of all geographical regions (A long list!!)
                              "restriction_level": ["internal", "confidential", "public"],
                              "language": None,  # TODO Read the list of ALL languages (or just "English"??)
                              }
                        if cl[keys[key]] and value.lower() not in cl[keys[key]]:
                            issues.append((3, "The key '"+key+"' should be one of: "+",".join(cl[keys[key]])))

                    if key not in content:
                        content[keys[key]] = []
                    content[keys[key]].append(str(value).strip())
        else:
            issues.append((2, "Row "+str(r)+": unknown metadata key '"+key+"'"))

    for key in keys:
        if mandatory[key] and keys[key] not in content:
            some_error = True
            issues.append((3, "The value '"+key+"' is mandatory in the definition of the metadata"))

    if True:
        return issues, None, content
    else:
        if not some_error:
            cmd = MetadataCommand(None)
            cmd.json_deserialize(content)
        else:
            cmd = None
        return cmd, issues


