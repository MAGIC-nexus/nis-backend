from backend.command_generators.spreadsheet_command_parsers_v2 import parse_command


def parse_hierarchy_command(sh, area, name: str):
    return parse_command(sh, area, name, "CatHierarchies")


def parse_hierarchy_mapping_command(sh, area, name: str):
    return parse_command(sh, area, name, "CatHierarchiesMapping")


def parse_parameters_command_v2(sh, area, name: str):
    return parse_command(sh, area, name, "Parameters")


def parse_contexts_command(sh, area, name: str):
    return parse_command(sh, area, name, "Contexts")
