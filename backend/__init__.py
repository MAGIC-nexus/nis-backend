import pint
from collections import namedtuple
from attr import attrs, attrib

# GLOBAL VARIABLES

# Database containing domain model and metadata about datasets
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
ureg.define("euro = [] = EUR = Eur = eur = Euro = Euros ")
ureg.define("dollar = [] = USD = Usd = usd = Dollar = Dollars")

# Named tuples
Issue = namedtuple("Issue",
                   "sheet_number sheet_name c_type type message")  # (Sheet #, Sheet name, command type, issue type, message)

SDMXConcept = namedtuple('Concept', 'type name istime description code_list')

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

# ##################################
# Commands


@attrs
class CommandField:
    # Allowed names for the column
    allowed_names = attrib()  # type: list[str]
    # Internal name used during the parsing
    name = attrib()  # type: str
    # Flag indicating if the column is mandatory or optional
    mandatory = attrib()  # type: bool
    # Parser for the column
    parser = attrib()
    # Some columns have a predefined set of allowed strings
    allowed_values = attrib(default=None)  # type: list[str]
    # Many values or just one
    many_values = attrib(default=True)
    # Many appearances (the field can appear multiple times). A convenience to define a list
    many_appearances = attrib(default=False)
    # Compiled regex
    regex_allowed_names = attrib(default=None)


