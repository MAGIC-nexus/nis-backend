"""
  Given one of the output datasets in State, convert it to SDMX format
  * Metadata
  * Data

  * pandaSDMX is only for reading. It has not been updated since two years
  * Eurostat has an IT tools page: https://ec.europa.eu/eurostat/web/sdmx-infospace/sdmx-it-tools
    * DSW (Data Structure Wizard). To create and maintain DSDs
    * SDMX Converter. Provides a web service to convert data to SDMX-ML. Input DSD is needed.
  * SdmxSource (http://www.sdmxsource.org/) is reference implementation now. Also two years without updates (last 9th June 2016)

DSPL (DataSet Publishing Language) from Google provides a Python package. "SDMX Converter" is able to use this.
"""