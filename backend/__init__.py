import pint

from collections import namedtuple

# GLOBAL VARIABLES

# Database containing domain model and metadata about datasets
engine = None

# Database containing OLAP data (cache of Data Cubes)
data_engine = None

# Data source manager
data_source_manager = None

# REDIS
redis = None

# Case sensitive
case_sensitive = False

# Create units registry
ureg = pint.UnitRegistry()
ureg.define("cubic_meter = m^3 = m3")

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

# Fields: ("<field label in excel file>", "<field name in Dublin Core>", Mandatory?, Controlled?, NameInJSON)
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
