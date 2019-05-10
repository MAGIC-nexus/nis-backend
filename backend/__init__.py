import configparser
import importlib
from enum import Enum
import os
import regex as re
from typing import Optional, Any, List, Tuple, Callable, Dict, Union, Type

import pint
from collections import namedtuple
from attr import attrs, attrib

# GLOBAL VARIABLES

engine = None

# Database containing OLAP data (cache of Data Cubes)
data_engine = None

# Data source manager
data_source_manager = None  # type: DataSourceManager

# REDIS
redis = None

# Case sensitive
case_sensitive = False

# Create units registry
ureg = pint.UnitRegistry()
ureg.define("cubic_meter = m^3 = m3")
ureg.define("euro = [] = EUR = Eur = eur = Euro = Euros = €")
ureg.define("dollar = [] = USD = Usd = usd = Dollar = Dollars = $")

# Named tuples
Issue = namedtuple("Issue",
                   "sheet_number sheet_name c_type type message")

SDMXConcept = namedtuple('Concept', 'type name istime description code_list')

# Global Types

IssuesOutputPairType = Tuple[List[Issue], Optional[Any]]
ExecutableCommandIssuesPairType = Tuple[Optional["IExecutableCommand"], List[Issue]]
IssuesLabelContentTripleType = Tuple[List[Issue], Optional[Any], Optional[Dict[str, Any]]]
# Tuple (top, bottom, left, right) representing the rectangular area of the input worksheet where the command is present
AreaTupleType = Tuple[int, int, int, int]

# ##################################
# METADATA special variables

# Simple DC fields not covered:
#  type (controlled),
#  format (controlled),
#  rights (controlled),
#  publisher,
#  contributor,
#  relation
#
# XML Dublin Core: http://www.dublincore.org/documents/dc-xml-guidelines/
# Exhaustive list: http://dublincore.org/documents/dcmi-type-vocabulary/

# Fields: ("<field label in Spreadsheet file>", "<field name in Dublin Core>", Mandatory?, Controlled?, NameInJSON)
metadata_fields = [("Case study name", "title", False, False, "case_study_name"),
                   ("Case study code", "title", True, False, "case_study_code"),
                   ("Title", "title", True, False, "title"),
                   ("Subject, topic and/or keywords", "subject", False, True, "subject_topic_keywords"),
                   ("Description", "description", False, False, "description"),
                   ("Geographical level", "description", True, True, "geographical_level"),
                   ("Dimensions", "subject", True, True, "dimensions"),
                   ("Reference documentation", "source", False, False, "reference_documentation"),
                   ("Authors", "creator", True, False, "authors"),
                   ("Date of elaboration", "date", True, False, "date_of_elaboration"),
                   ("Temporal situation", "coverage", True, False, "temporal_situation"),
                   ("Geographical location", "coverage", True, False, "geographical_situation"),
                   ("DOI", "identifier", False, False, "doi"),
                   ("Language", "language", True, True, "language"),
                   ("Restriction level", None, True, True, "restriction_level"),
                   ("Version", None, True, False, "version")
                   ]

# Regular expression definitions
regex_var_name = "([a-zA-Z][a-zA-Z0-9_-]*)"
regex_hvar_name = "(" + regex_var_name + r"(\." + regex_var_name + ")*)"
regex_cplex_var = "((" + regex_var_name + "::)?" + regex_hvar_name + ")"
regex_optional_alphanumeric = "([ a-zA-Z0-9_-]*)?"  # Whitespace also allowed


# Regular expression for "worksheet name" in version 2
def simple_regex(names: List[str]):
    return r"(" + "|".join(names) + ")" + regex_optional_alphanumeric


# Global configuration variables
global_configuration = None


def get_global_configuration_variable(key: str) -> str:
    def read_configuration() -> Dict[str, str]:
        """
        If environment variable "MAGIC_NIS_SERVICE_CONFIG_FILE" is defined, and the contents is the name of an existing file,
        read it as a configuration file and return the result

        :return:
        """
        if os.environ.get("MAGIC_NIS_SERVICE_CONFIG_FILE"):
            fname = os.environ["MAGIC_NIS_SERVICE_CONFIG_FILE"]
            if os.path.exists(fname):
                with open(fname, 'r') as f:
                    config_string = '[asection]\n' + f.read()
                config = configparser.ConfigParser()
                config.read_string(config_string)
                return {t[0]: t[1] for t in config.items("asection")}
            else:
                return {}
        else:
            return {}

    global global_configuration
    if global_configuration is None:
        global_configuration = read_configuration()
    return global_configuration.get(key)

# ##################################
# Commands


class CommandType(Enum):
    input = (1, "Input")
    core = (2, "Core")
    convenience = (3, "Convenience")
    metadata = (4, "Metadata")
    analysis = (5, "Analysis")
    misc = (99, "Miscellaneous")

    def __str__(self):
        return self.value[1]

    @classmethod
    def from_string(cls, s):
        for ct in cls:
            if ct.value[1] == s:
                return ct
        raise ValueError(cls.__name__ + ' has no value matching "' + s + '"')


@attrs(cmp=False)  # Constant and Hashable by id
class Command:
    # Name, the lowercase unique name
    name = attrib()  # type: str
    # Allowed names for the worksheet. Used for simple regular expressions.
    allowed_names = attrib()  # type: List[str]
    # Name of the subclass of IExecutableCommand in charge of the execution
    execution_class_name = attrib()  # type: Optional[str]
    # Command type
    cmd_type = attrib()  # type: CommandType
    # Direct examples
    direct_examples = attrib(default=[])  # type: Optional[List[str]]
    # URLs of files where it is used
    files = attrib(default=[])  # type: Optional[List[str]]
    # Alternative regular expression for worksheet name, otherwise the simple_regex() is used
    alt_regex = attrib(default=None)
    # Parse function, having params (Worksheet, Area) and returning a tuple (issues, label, content)
    # Callable[[Worksheet, AreaTupleType, str, ...], IssuesLabelContentTripleType] = attrib(default=None)
    parse_function: Callable[..., IssuesLabelContentTripleType] = attrib(default=None)
    # In which version is this command allowed?
    is_v1 = attrib(default=False)  # type: bool
    is_v2 = attrib(default=False)  # type: bool

    @property
    def regex(self):
        if self.alt_regex:
            pattern = self.alt_regex
        else:
            pattern = simple_regex(self.allowed_names)

        return re.compile(pattern, flags=re.IGNORECASE)

    @property
    def execution_class(self):
        if self.execution_class_name:
            module_name, class_name = self.execution_class_name.rsplit(".", 1)
            return getattr(importlib.import_module(module_name), class_name)
        else:
            return None


@attrs(cmp=False)  # Constant and Hashable by id
class CommandField:
    # Allowed names for the column
    allowed_names = attrib()  # type: List[str]
    # Internal name used during the parsing
    name = attrib()  # type: str
    # Parser for the column
    parser = attrib()
    # Flag indicating if the column is mandatory or optional. It can also be an expression (string).
    mandatory = attrib(default=False)  # type: Union[bool, str]
    # A default value for the field
    default_value = attrib(default=None)
    # Some columns have a predefined set of allowed strings
    allowed_values = attrib(default=None)  # type: Optional[list[str]]
    # Many values or just one
    many_values = attrib(default=True)
    # Many appearances (the field can appear multiple times). A convenience to define a list
    many_appearances = attrib(default=False)
    # Examples
    examples = attrib(default=None)  # type: List[str]
    # Compiled regex
    # regex_allowed_names = attrib(default=None)
    # Is it directly an attribute of a Musiasem type? Which one?
    attribute_of = attrib(default=None)  # type: Type

    @property
    def regex_allowed_names(self):
        def contains_any(s, setc):
            return 1 in [c in s for c in setc]

        # Compile the regular expressions of column names
        rep = [(r if contains_any(r, ".+") else re.escape(r))+"$" for r in self.allowed_names]

        return re.compile("|".join(rep), flags=re.IGNORECASE)
