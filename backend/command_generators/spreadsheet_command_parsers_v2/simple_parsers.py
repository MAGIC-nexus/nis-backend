from backend.command_generators.spreadsheet_command_parsers_v2 import parse_command


# "DatasetData" needs a specialized parser
# "DatasetQry" needs a specialized parser
# "ProblemStatement" needs a specialized parser
# "Metadata" needs a specialized parser

def parse_cat_hierarchy_command(sh, area, name: str=None):
    return parse_command(sh, area, name, "CatHierarchies")


def parse_hierarchy_mapping_command(sh, area, name: str=None):
    return parse_command(sh, area, name, "CatHierarchiesMapping")


def parse_parameters_command_v2(sh, area, name: str=None):
    return parse_command(sh, area, name, "Parameters")


def parse_attribute_sets_command(sh, area, name: str=None):
    return parse_command(sh, area, name, "AttributeSets")


def parse_attribute_types_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "AttributeTypes")


def parse_datasetdef_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "DatasetDef")


def parse_parameters_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "Parameters")


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


def parse_indicators_v2_command(sh, area, name: str = None):
    return parse_command(sh, area, name, "Indicators")
