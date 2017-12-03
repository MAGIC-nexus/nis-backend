import pint

# GLOBAL VARIABLES

# Database containing domain model and metadata about datasets
engine = None

# Database containing OLAP data (cache of Data Cubes)
data_engine = None

# Create units registry
ureg = pint.UnitRegistry()
