import regex as re
from uncertainties import ufloat_fromstr
from pint.errors import UndefinedUnitError

from backend.command_generators.basic_elements_parser import simple_ident
from backend.common.helper import create_dictionary, case_sensitive
from backend import ureg
from backend.command_generators import basic_elements_parser
from backend.model.memory.musiasem_concepts import allowed_ff_types

"""
User Interface. There will be a special page for command_executors reference.
 * Popup. Search. Copy, paste into the Worksheet
 * Submit command
User Interface. how navigation works. Is it compatible with editors like "Bootstrap Studio", "PineWorks", "Pingendo"?
User Interface. there will be a special page to browse Datasets and their input parameters

All command_executors store the content as JSON loads or similar. Use that for the "serialize" command.
 
* Transform columns into processors with observations
* Need JSONLink to create pointers:
  * Each object needs an ID
  * Pointers point to IDs
  * Dictionary from ID to Location of JSON

* JSONSchema to validate that the structure generated internally is correct
* A graph of Observables
* Observables having Observations
* CALCULATED OBSERVATIONS. Check existence. If not, look for a formula or set of rules matching the requested Observation
  * Then, there must be a repository of rules
* A structure where it is easy to ADD new observations

OUTPUT COMMANDS
* Workbook: create workbook. Pivot table, processors.
* Graph format:  
* Online: browse pivot table, processor relations
"""


def parse_data_input_command(sh, area, processors_type, state):
    """
    Scans the "area" of input worksheet "sh" where it is assumed a "data input" command
    is present.

    It obtains a list of observations, a list of processors, a list of observables, a list of tags
    All those are represented in JSON format

    :param sh: Input worksheet
    :param area: Tuple (top, bottom, left, right) representing the rectangular area of the input worksheet where the
    command is present
    :param processors_type: Name for the type of processors. Also label of the command
    :param state: Transient state useful for checking existence of variables
    :return: DataInputCommand, list of issues (issue_type, message)
    """
    some_error = False
    issues = []
    # Define a set of observations (qualified quantities) of observables
    # This set can be replicated. So, ?how to refer to each replica?
    # Regular expression, internal name, Mandatory (True|False)
    known_columns = [(r"Name|Processor[_ ]name", "processor", False),
                     (r"Level", "level", False),
                     (r"Parent", "parent", False),
                     (r"FF[_ ]type", "ff_type", False),
                     (r"Var|Variable", "factor", True),
                     (r"Value|NUSAP\.N", "value", True),
                     (r"Unit|NUSAP\.U", "unit", True),
                     (r"Relative[_ ]to", "relative_to", False),
                     (r"Uncertainty|Spread|NUSAP\.S", "uncertainty", False),
                     (r"Assesment|NUSAP\.A", "assessment", False),
                     (r"Pedi[_ ]template|NUSAP\.PT", "pedigree_template", False),
                     (r"Pedigree|NUSAP\.P", "pedigree", False),
                     (r"Time", "time", False),
                     (r"Geo|Geolocation", "geolocation", False),
                     (r"Source", "source", False),
                     (r"Comment|Comments", "comments", False)
                     ]

    # First, examine columns, to know which fields are being specified
    # Special cases:
    #   Open columns: the field is specified in the cell togheter with the value. Like "attr1=whatever", instead of a header "attr1" and in a row below, a value "whatever"
    #   Complex values: the value has syntactic rules. Like expressions for both quantities AND qualities (like NUSAP)
    #   References: the field refers to additional information in another worksheet. Unique names or ref holder (worksheet name) plus ref inside the worksheet, would be allowed. Also ref type can disambiguate
    mandatory = {t[1]: t[2] for t in known_columns}
    cre = {}  # Column Regular Expression dictionary (K: regular expression; V: RegularExpression object)
    if not case_sensitive:
        flags = re.IGNORECASE
    else:
        flags = 0
    for kc in known_columns:
        cre[kc[0]] = re.compile(kc[0], flags=flags)
    col_names = {}
    standard_cols = {}  # Internal (standardized) column name to column index in the worksheet (freedom in the order of columns)
    attribute_cols = create_dictionary()  # Not recognized columns are considered freely named categories, attributes or tags
    attributes = []  # List of attributes or tags (keys of the previous dictionary)
    col_allows_dataset = create_dictionary()  # If the column allows the reference to a dataset dimension
    for c in range(area[2], area[3]):
        col_name = sh.cell(row=area[0], column=c).value
        if not col_name:
            continue

        col_names[c] = col_name

        # Match
        found = False
        for kc in known_columns:
            res = cre[kc[0]].search(col_name)
            if res:
                if kc[1] in standard_cols:
                    issues.append((2, "Cannot repeat column name '" + col_name +"' (" + kc[0] +") in data input command '" + processors_type + "'"))
                else:
                    standard_cols[kc[1]] = c
                    col_names[c] = kc[1]  # Override column name with pseudo column name for standard columns
                    if col_names[c].lower() in ["factor", "value", "time", "geolocation"]:
                        col_allows_dataset[col_names[c]] = True
                    else:
                        col_allows_dataset[col_names[c]] = False
                    found = True
                break
        if not found:
            if col_name not in attribute_cols:
                # TODO Check valid col_names. It must be a valid Variable Name
                attribute_cols[col_name] = c
                attributes.append(col_name)
                col_allows_dataset[col_name] = True
            else:
                issues.append((2, "Cannot repeat column name '" + col_name + "' in data input command '" + processors_type + "'"))

    del cre

    # Check if there are mandatory columns missing

    # TODO There could be combinations of columns which change the character of mandatory of some columns
    # TODO For instance, if we are only specifying structure, Value would not be needed
    for kc in known_columns:
        # "kc[2]" is the flag indicating if the column is mandatory or not
        # col_map contains standard column names present in the worksheet
        if kc[2] and kc[1] not in standard_cols:
            some_error = True
            issues.append((3, "Column name '" + kc[0] +"' must be specified in data input command '" + processors_type + "'"))

    # If there are errors, do not continue
    if some_error:
        return None, issues

    processor_attribute_exclusions = create_dictionary()
    processor_attribute_exclusions["scale"] = None  # Exclude these attributes when characterizing the processor
    processor_attributes = [t for t in attributes if t not in processor_attribute_exclusions]

    # SCAN rows
    lst_observations = []  # List of ALL observations. -- Main outcome of the parse operation --

    set_pedigree_templates = create_dictionary()  # List of pedigree templates
    set_processors = create_dictionary()  # List of processor names
    set_factors = create_dictionary()  # List of factors
    set_taxa = create_dictionary()  # Dictionary of taxa with their lists of values. Useful to return CODE LISTS
    set_referenced_datasets = create_dictionary()  # Dictionary of datasets to be embedded into the result (it is a job of the execution part)
    processors_taxa = create_dictionary()  # Correspondence "processor" -> taxa (to avoid changes in this correspondence)

    dataset_column_rule = basic_elements_parser.dataset_with_column
    values = [None]*area[3]
    # LOOP OVER EACH ROW
    for r in range(area[0] + 1, area[1]):  # Scan rows (observations)
        # Each row can specify: the processor, the factor, the quantity and qualities about the factor in the processor
        #                       It can also specify a "flow+containment hierarchy" relation

        row = {}  # Store parsed values of the row

        taxa = create_dictionary()  # Store attributes or taxa of the row

        referenced_dataset = None  # Once defined in a row, it cannot change!!
        # Scan the row first, looking for the dataset. The specification is allowed in certain columns:
        # attribute_cols and some standard_cols
        already_processed = create_dictionary()
        for c in range(area[2], area[3]):
            if c in col_names:
                value = sh.cell(row=r, column=c).value
                if isinstance(value, str) and value.startswith("#"):
                    col_name = col_names[c]
                    if col_allows_dataset[col_name]:
                        if not referenced_dataset:
                            try:
                                ast = basic_elements_parser.string_to_ast(dataset_column_rule, value[1:])
                                if len(ast["parts"]) == 2:
                                    referenced_dataset = ast["parts"][0]
                                    # Remove the dataset variable. It will be stored in "_referenced_dataset"
                                    value = "#" + ast["parts"][1]
                                else:
                                    some_error = True
                                    issues.append((3, "The first dataset reference of the row must contain the "
                                                      "dataset variable name and the dimension name, row " + str(r)))

                                # Mark as processed
                                already_processed[col_name] = None
                            except:
                                some_error = True
                                issues.append((3, "Column '" + col_name + "' has an invalid dataset reference '" + value + "', in row " + str(r)))
                        else:
                            try:
                                ast = basic_elements_parser.string_to_ast(simple_ident, value[1:])
                                # Mark as processed
                                already_processed[col_name] = None
                            except:
                                some_error = True
                                issues.append((3, "Column '" + col_name + "' has an invalid dataset reference '" + value + "', in row " + str(r)))
                        if col_name in standard_cols:
                            row[col_name] = value
                        else:
                            taxa[col_name] = value

                values[c] = value

        # TODO If the flow type is decomposed, compose it first
        for c in standard_cols:
            if c in already_processed:
                continue

            value = values[standard_cols[c]]

            # != "" or not
            if value is None or (value is not None and value == ""):
                if mandatory[c]:
                    some_error = True
                    issues.append((3, "Column '" + c + "' is mandatory, row " + str(r)))
                continue  # Skip the rest of the iteration!

            # Parse the value
            if c in ["processor", "factor", "pedigree_template"]:
                # Check that it is a variable name, and allow hierarchical names
                basic_elements_parser.string_to_ast(basic_elements_parser.simple_h_name, value)
            elif c == "relative_to":
                # Two elements, the first a hierarchical name, the second a unit name
                s = value.split(" ")
                if len(s) != 2:
                    some_error = True
                    issues.append((3, "The Relative To value has to have two parts, factor name and unit, separated by a whitespace (specified '"+value+"'), in row "+str(r)))
                else:
                    try:
                        basic_elements_parser.string_to_ast(basic_elements_parser.simple_h_name, s[0])
                    except:
                        some_error = True
                        issues.append((3, "The name specified for the relative to factor '"+s[0]+"' is not valid, in row "+str(r)))

                    # It must be a recognized unit. Check with Pint
                    try:
                        ureg(s[1])
                        ureg.parse_unit_name(s[1], case_sensitive)
                    except UndefinedUnitError:
                        some_error = True
                        issues.append((3, "The unit name '"+s[1]+"' is not registered in the units processing package, in row "+str(r)))
            elif c == "level":
                # A valid level name
                basic_elements_parser.string_to_ast(basic_elements_parser.level_name, value)
            elif c == "parent":
                # Check that value is a valid parent name. It can be either a list of tags OR
                # a processor name, something defining a single processor
                try:
                    basic_elements_parser.string_to_ast(basic_elements_parser.simple_h_name, value)
                except:
                    try:
                        basic_elements_parser.string_to_ast(basic_elements_parser.named_parameters_list, value)
                    except:
                        some_error = True
                        issues.append((3, "Could not parse '"+value+"' as 'parent' in row " + str(r)))
            elif c == "ff_type":
                # The type of flow/fund must be one of a set of possible values. DEFINE THE LIST
                if value.lower() not in allowed_ff_types:
                    some_error = True
                    issues.append((3, "ff_type must be one of :"+', '.join(allowed_ff_types)+", in row " + str(r)))
            elif c == "value":
                if not isinstance(value, str):
                    value = str(value)
                # Expression allowed. Check syntax only. It can refer to parameters.
                ast = basic_elements_parser.string_to_ast(basic_elements_parser.expression, value)
                # TODO Check existence of used variables
                # TODO basic_elements_parser.ast_evaluator(ast, state, None, issues, "static")
            elif c == "unit":
                # It must be a recognized unit. Check with Pint
                try:
                    if value == "-":
                        value = ""  # Dimensionless
                    ureg(value)
                    ureg.parse_unit_name(value, case_sensitive)
                except UndefinedUnitError:
                    some_error = True
                    issues.append((3, "The unit name '"+value+"' is not registered in the units processing package, in row "+str(r)))
            elif c == "uncertainty":
                # TODO It must be a valid uncertainty specificator
                pass
            elif c == "assessment":
                # A free text
                pass
            elif c == "pedigree":
                # TODO A valid pedigree specification
                pass
            elif c == "time":
                # A valid time specification. Possibilities: Year, Month-Year / Year-Month, Time span (two dates)
                if not isinstance(value, str):
                    value = str(value)
                ast = basic_elements_parser.string_to_ast(basic_elements_parser.time_expression, value)
            elif c == "geolocation":
                # A valid GEO specification
                # ast = basic_elements_parser.string_to_ast(basic_elements_parser.geo_expression, value)
                pass
            elif c == "source":
                # TODO Who or what provided the information. It can be formal or informal. Formal can be references
                pass
            elif c == "comments":
                # Free text
                pass

            # Store the parsed value
            row[c] = value

        for c in attribute_cols:
            if c in already_processed:
                continue

            value = values[attribute_cols[c]]

            # != "" or not
            if not value:
                taxa[c] = None
                continue  # Skip the rest of the iteration!

            # TODO Check value. Valid identifier, no whitespace
            # Validate "value", it has to be a simple ID
            try:
                if not isinstance(value, str):
                    value = str(value)
                basic_elements_parser.simple_ident.parseString(value, parseAll=True)
            except:
                value = None
                some_error = True
                issues.append((3,
                               "The value in column '" + c + "' has to be a simple identifier: start with letter, then letters, numbers and '_', no whitespace, in row " + str(
                                   r)))

            taxa[c] = value

            # Disable the registration of taxa. If a Dataset reference is used, there is no way to register
            # taxa at parse time (the dataset is still not obtained). Leave it for the execution
            if c not in set_taxa:
                set_taxa[c] = create_dictionary()
            set_taxa[c][value] = None

        # Now that individual columns have been parsed, do other things

        if referenced_dataset:
            row["_referenced_dataset"] = referenced_dataset

        # If "processor" not specified, concatenate taxa columns in order to generate an automatic name
        # (excluding the processor type)
        p_taxa = taxa.copy()
        for k in processor_attribute_exclusions:
            if k in p_taxa: del p_taxa[k]

        if "processor" not in row:
            row["processor"] = "_".join([str(taxa[t]) for t in processor_attributes])  # TODO Which order? (the current is "order of appearance"; maybe "alphabetical order" would be better option)
        # Add as "taxa" the processor type (which is an optional input parameter to this function)
        if processors_type:
            taxa["_processors_type"] = processors_type
        # Store taxa (attributes and taxa)
        row["taxa"] = taxa
        # Store taxa if the processor still does not have it
        if row["processor"] not in processors_taxa:
            processors_taxa[row["processor"]] = p_taxa  # "::".join([taxa[t] for t in lst_taxa_cols])
        else:
            # Taxa should be the same for each "processor". Error if different
            t = processors_taxa[row["processor"]]
            if t != p_taxa:
                issues.append((3, "The processor '"+row["processor"]+"' has different taxa assigned, in row "+str(r)))

        # Register new processor names, pedigree templates, and variable names
        if "processor" in row:
            set_processors[row["processor"]] = None
        if "pedigree_template" in row:
            set_pedigree_templates[row["pedigree_template"]] = None
        if "factor" in row:
            set_factors[row["factor"]] = None
        if referenced_dataset:
            set_referenced_datasets[referenced_dataset] = None

        lst_observations.append(row)

    label = "Processors " + processors_type
    content = {"factor_observations": lst_observations,
               "processor_attributes": processor_attributes,
               "processors": [k for k in set_processors],
               "pedigree_templates": [k for k in set_pedigree_templates],
               "factors": [k for k in set_factors],
               "referenced_datasets": [ds for ds in set_referenced_datasets],
               "code_lists": {k: [k2 for k2 in set_taxa[k]] for k in set_taxa}
               }
    return issues, label, content
    # if not some_error:
    #     cmd = DataInputCommand(label)
    #     cmd.json_deserialize(content)
    # else:
    #     cmd = None
    # return cmd, issues
