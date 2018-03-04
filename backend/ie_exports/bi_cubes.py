"""
  Given a dataset:
    * prepare a "schema.xml" record (Mondrian specific)
    * prepare a relational database and load into it

  The dataset uses the dataset model: Dataset, Dimension, Codelist for metadata and pd.DataFrame for the data

  The preparation of tables in the relational database overwrites existing
  The schema.xml
"""

