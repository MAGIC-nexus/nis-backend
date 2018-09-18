from backend.command_generators.spreadsheet_command_parsers_v2 import parse_command


# "DatasetData" needs a specialized parser
# "DatasetQry" needs a specialized parser
# "ProblemStatement" needs a specialized parser
# "Metadata" needs a specialized parser

def parse_cat_hierarchy_command(sh, area, name: str=None):
    return parse_command(sh, area, name, "CatHierarchies")


def parse_hierarchy_mapping_command(sh, area, name: str=None):
    return parse_command(sh, area, name, "CatHierarchiesMapping")


def parse_attribute_types_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "AttributeTypes")


def parse_datasetdef_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "DatasetDef")

# NOTE: "DatasetData" is a special parser. Column names correspond to concept names: dimensions, measures or attributes


def parse_attribute_sets_command(sh, area, name: str=None):
    return parse_command(sh, area, name, "AttributeSets")


def parse_parameters_command_v2(sh, area, name: str=None):
    return parse_command(sh, area, name, "Parameters")

# NOTE: "DatasetQry" is a special parser. Column names may have dynamic meaning if they are concept names (from dataset)


def parse_interface_types_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "InterfaceTypes")


def parse_processors_v2_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "Processors")


def parse_interfaces_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "Interfaces")


def parse_relationships_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "Relationships")


def parse_instantiations_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "Instantiations")


def parse_scale_changers_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "ScaleChangers")


def parse_shared_elements_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "SharedElements")


def parse_reused_elements_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "ReusedElements")

# NOTE: Metadata is special parser
# NOTE: PedigreeMatrices and References are special parsers


def parse_indicators_v2_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "Indicators")

# NOTE: Other commands may come here: Problem Specification, Dashboards (or the new Storyboards) which include defining
# NOTE: each of the views in it.
