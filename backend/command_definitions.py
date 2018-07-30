import re

from backend import CommandField
from backend.command_generators.basic_elements_parser import simple_ident, unquoted_string, alphanums_string, \
    hierarchy_expression_v2, key_value_list, key_value, parameter_value, parameter_domain, expression_with_parameters


commands = {
    "CatHierarchies":
    [CommandField(allowed_names=["Source"], name="source", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["HierarchyGroup"], name="hierarchy_group", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["HierarchyName"], name="hierarchy_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Level", "LevelCode"], name="level", mandatory=False, allowed_values=None, parser=alphanums_string),
     CommandField(allowed_names=["ReferredHierarchy"], name="referred_hierarchy", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Code"], name="code", mandatory=False, allowed_values=None, parser=alphanums_string),
     # NOTE: Removed because parent code must be members of the same hierarchy in definition
     # CommandField(allowed_names=["ReferredHierarchyParent"], name="referred_hierarchy_parent", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["ParentCode"], name="parent_code", mandatory=False, allowed_values=None, parser=alphanums_string),
     CommandField(allowed_names=["Label"], name="label", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Expression", "Formula"], name="expression", mandatory=False, allowed_values=None, parser=hierarchy_expression_v2),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["Attribute"], name="attribute", mandatory=False, allowed_values=None, parser=key_value)
     ],
    "CatHierarchiesMapping":
    [CommandField(allowed_names=["SourceHierarchy"], name="source_hierarchy", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["SourceCode"], name="source_code", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DestinationHierarchy"], name="destination_hierarchy", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DestinationCode"], name="destination_hierarchy", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Weight"], name="weight", mandatory=True, allowed_values=None, parser=expression_with_parameters)
     ],
    "Parameters":
    [CommandField(allowed_names=["Name", "ParameterName"], name="name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Type"], name="type", mandatory=True, allowed_values=["Number", "Category", "Boolean"], parser=simple_ident),
     CommandField(allowed_names=["Domain"], name="domain", mandatory=False, allowed_values=None, parser=parameter_domain),
     CommandField(allowed_names=["Value"], name="value", mandatory=False, allowed_values=None, parser=expression_with_parameters),
     CommandField(allowed_names=["Description"], name="description", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["Attribute"], name="attribute", mandatory=False, allowed_values=None, parser=key_value)
     ],
    "Contexts":
    [CommandField(allowed_names=["ContextName"], name="context_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["Attribute"], name="attribute", mandatory=False, allowed_values=None, parser=key_value)
     ]
}


def compile_command_field_regexes():
    for cmd in commands:
        # Compile the regular expressions of column names
        flags = re.IGNORECASE
        for c in commands[cmd]:
            rep = [re.escape(r) for r in c.allowed_names]
            c.regex_allowed_names = re.compile("|".join(rep), flags=flags)


compile_command_field_regexes()
