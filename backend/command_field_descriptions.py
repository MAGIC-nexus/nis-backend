
"""
    "scale_conversion_v2": [
        CommandField(allowed_names=["OriginHierarchy"], name="source_hierarchy", parser=simple_ident),
        CommandField(allowed_names=["OriginInterfaceType"], name="source_interface_type", mandatory=True, parser=simple_ident),
        CommandField(allowed_names=["DestinationHierarchy"], name="target_hierarchy", parser=simple_ident),
        CommandField(allowed_names=["DestinationInterfaceType"], name="target_interface_type", mandatory=True, parser=simple_ident),
        CommandField(allowed_names=["OriginContext"], name="source_context", parser=processor_names),
        CommandField(allowed_names=["DestinationContext"], name="target_context", parser=processor_names),
        CommandField(allowed_names=["Scale"], name="scale", mandatory=True, parser=expression_with_parameters),
        CommandField(allowed_names=["OriginUnit"], name="source_unit", parser=unit_name),
        CommandField(allowed_names=["DestinationUnit"], name="target_unit", parser=unit_name)
    ],

    "import_commands": [
        CommandField(allowed_names=["Workbook", "WorkbookLocation"], name="workbook_name", parser=url_parser),
        CommandField(allowed_names=["Worksheets"], name="worksheets", parser=unquoted_string)
    ],

    "list_of_commands": [
        CommandField(allowed_names=["Worksheet", "WorksheetName"], name="worksheet", mandatory=True, parser=unquoted_string),
        CommandField(allowed_names=["Command"], name="command", mandatory=True, allowed_values=valid_v2_command_names, parser=simple_ident),
        CommandField(allowed_names=["Comment", "Description"], name="comment", parser=unquoted_string)
    ],

    "ref_provenance": [
        # Reduced, from W3C Provenance Recommendation (https://www.w3.org/TR/prov-overview/)
        CommandField(allowed_names=["RefID", "Reference"], name="ref_id", mandatory=True, parser=simple_ident),
        # The reference "RefID" should be mentioned
        CommandField(allowed_names=["ProvenanceFileURL"], name="provenance_file_url", parser=url_parser),
        CommandField(allowed_names=["AgentType"], name="agent_type", mandatory=True, allowed_values=agent_types, parser=simple_ident),
        CommandField(allowed_names=["Agent"], name="agent", mandatory=True, parser=unquoted_string),
        CommandField(allowed_names=["Activities"], name="activities", mandatory=True, parser=unquoted_string),
        CommandField(allowed_names=["Entities"], name="entities", parser=unquoted_string)
    ],

    "ref_geographical": [
        # A subset of fields from INSPIRE regulation for metadata: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32008R1205&from=EN
        # Fields useful to elaborate graphical displays. Augment in the future as demanded
        CommandField(allowed_names=["RefID", "Reference"], name="ref_id", mandatory=True, parser=simple_ident),
        CommandField(allowed_names=["GeoLayerURL"], name="geo_layer_url", parser=url_parser),
        CommandField(allowed_names=["Title"], name="title", mandatory=True, parser=unquoted_string),
        CommandField(allowed_names=["Abstract"], name="abstract", parser=unquoted_string),  # Syntax??
        CommandField(allowed_names=["Type"], name="type", allowed_values=geographic_resource_types, parser=unquoted_string),  # Part D.1. JUST "Dataset"
        CommandField(allowed_names=["ResourceLocator", "DataLocation"], name="data_location", parser=url_parser),
        CommandField(allowed_names=["TopicCategory"], name="topic_category", allowed_values=geographic_topic_categories, parser=unquoted_string),  # Part D.2
        CommandField(allowed_names=["BoundingBox"], name="bounding_box", parser=unquoted_string),  # Syntax??
        CommandField(allowed_names=["TemporalExtent", "Date"], name="temporal_extent", parser=unquoted_string),  # Syntax??
        CommandField(allowed_names=["PointOfContact"], name="metadata_point_of_contact", parser=unquoted_string)
    ],

    "ref_bibliographic": [
        # From BibTex. Mandatory fields depending on EntryType, at "https://en.wikipedia.org/wiki/BibTeX" (or search: "Bibtex entry field types")
        CommandField(allowed_names=["RefID", "Reference"], name="ref_id", mandatory=True, parser=simple_ident),
        CommandField(allowed_names=["BibFileURL"], name="bib_file_url", parser=url_parser),
        CommandField(allowed_names=["EntryType"], name="entry_type", mandatory=True, allowed_values=bib_entry_types, parser=unquoted_string),
        CommandField(allowed_names=["Address"], name="address", parser=unquoted_string),
        CommandField(allowed_names=["Annote"], name="annote", parser=unquoted_string),
        CommandField(allowed_names=["Author"], name="author", mandatory="entry_type not in ('booklet', 'manual', 'misc', 'proceedings')", parser=unquoted_string),
        CommandField(allowed_names=["BookTitle"], name="booktitle", mandatory="entry_type in ('incollection', 'inproceedings')", parser=unquoted_string),
        CommandField(allowed_names=["Chapter"], name="chapter", mandatory="entry_type in ('inbook')", parser=unquoted_string),
        CommandField(allowed_names=["CrossRef"], name="crossref", parser=unquoted_string),
        CommandField(allowed_names=["Edition"], name="edition", parser=unquoted_string),
        CommandField(allowed_names=["Editor"], name="editor", mandatory="entry_type in ('book', 'inbook')", parser=unquoted_string),
        CommandField(allowed_names=["HowPublished"], name="how_published", parser=unquoted_string),
        CommandField(allowed_names=["Institution"], name="institution", mandatory="entry_type in ('techreport')", parser=unquoted_string),
        CommandField(allowed_names=["Journal"], name="journal", mandatory="entry_type in ('article')", parser=unquoted_string),
        CommandField(allowed_names=["Key"], name="key", parser=unquoted_string),
        CommandField(allowed_names=["Month"], name="month", allowed_values=bib_months, parser=simple_ident),
        CommandField(allowed_names=["Note"], name="note", parser=unquoted_string),
        CommandField(allowed_names=["Number"], name="number", parser=unquoted_string),
        CommandField(allowed_names=["Organization"], name="organization", parser=unquoted_string),
        CommandField(allowed_names=["Pages"], name="pages", mandatory="entry_type in ('inbook')", parser=unquoted_string),
        CommandField(allowed_names=["Publisher"], name="publisher", mandatory="entry_type in ('book', 'inbook', 'incollection')", parser=unquoted_string),
        CommandField(allowed_names=["School"], name="school", mandatory="entry_type in ('mastersthesis', 'phdtesis')", parser=unquoted_string),
        CommandField(allowed_names=["Series"], name="series", parser=unquoted_string),
        CommandField(allowed_names=["Title"], name="title", mandatory="entry_type not in ('misc')", parser=unquoted_string),
        CommandField(allowed_names=["Type"], name="type", parser=unquoted_string),
        CommandField(allowed_names=["URL"], name="url", parser=url_parser),
        CommandField(allowed_names=["Volume"], name="volume", mandatory="entry_type in ('article')", parser=unquoted_string),
        CommandField(allowed_names=["Year"], name="year", mandatory="entry_type in ('article', 'book', 'inbook', 'incollection', 'inproceedings', 'mastersthesis', 'phdthesis', 'proceedings', 'techreport')", parser=unquoted_string)
    ],

"""
cf_descriptions = {
    # CodeHierarchies
    ("cat_hierarchies", "source"): "Data producing source defining the hierarchy. For instance “Eurostat” or “FAO”.",
    ("cat_hierarchies", "hierarchy_group"): "Name of a group of hierarchies. Because hierarchies must be members of a HierarchyGroup, when left blank, the Hierarchy name is passed to the HierarchyGroup. At the same time, this means that the Hierarchy is a Code List which is also a base hierarchy, whose codes can be rearranged/reorganized by Hierarchies referring to them.",
    ("cat_hierarchies", "hierarchy_name"): "Name of a hierarchy. The name must be unique inside the group.",
    ("cat_hierarchies", "level"): "Name of a level in a hierarchy. Codes in a Hierarchy can be grouped in levels, this field allows specifying this.",
    ("cat_hierarchies", "referred_hierarchy"): "Name of the base Hierarchy -Code List- where the code is defined originally. Optional when defining a base Hierarchy -Code List-, mandatory when defining Hierarchical Code List",
    ("cat_hierarchies", "code"): "The formal name of a node in the hierarchy. This name can be used later in different places: the creation of hierarchical code lists, values in attributes, expressions...",
    ("cat_hierarchies", "parent_code"): "If the code being defined does not have parent the value is empty, if not, specify the parent Code.",
    ("cat_hierarchies", "label"): "A label that will be used when elaborating datasets or visualizations.",
    ("cat_hierarchies", "description"): "A free description optionally used for readability.",
    ("cat_hierarchies", "expression"): "A simple arithmetic expression referring to codes in the same hierarchy. No circular references allowed.",
    ("cat_hierarchies", "geolocation_ref"): "The ID of a 'geographic dataset' reference.",
    ("cat_hierarchies", "geolocation_code"): "When “Code” does not identify a feature in the referred dataset, this field can be used. ",
    ("cat_hierarchies", "attributes"): "A list of freely definable attributes attached to the code. Columns with a header starting in “@” are considered attribute columns that are appended to the list of attributes.",
    # CodeHierarchiesMapping
    ("cat_hier_mapping", "source_dataset"): "Dataset containing the definition of the hierarchy (for hierarchies defined through importing datasets).",
    ("cat_hier_mapping", "source_hierarchy"): "Name of the origin Hierarchy.",
    ("cat_hier_mapping", "source_code"): "Name of a code in the origin Hierarchy.",
    ("cat_hier_mapping", "destination_hierarchy"): "Name of the destination Hierarchy.",
    ("cat_hier_mapping", "destination_code"): "Name of a code in the destination Hierarchy.",
    ("cat_hier_mapping", "weight"): "Weight permits splitting an origin code in more than one destination codes, or simply change the scale of an origin code into the destination code. If a code is split the sum should not be more than one. Expressions using parameters are permitted. In case “weight” is left empty, it is equivalent to specifying “1”.",
    # DatasetDef
    ("datasetdef", "dataset_name"): "Name of the custom dataset.",
    ("datasetdef", "dataset_data_location"): "Location of data of this dataset. If data is embedded in the same spreadsheet (DatasetData command), “data://” must be used. If not, the location of a publicly available CSV or XLSX file must be provided, using a URL. If a XLSX file is used, the worksheet name to be used is specified appending to the URL the character “#” and the name of the worksheet. If the worksheet is local, the syntax “data://#” followed by the “DatasetData” worksheet name can be used.",
    ("datasetdef", "concept_type"): "Empty for the Dataset header row, One of the allowed values to define a concept",
    ("datasetdef", "concept_name"): "Name of the Concept being defined.",
    ("datasetdef", "concept_data_type"): "Data-type of the Concept.",
    ("datasetdef", "concept_domain"): "Allows constraining the values of a concept. Dimensions typically are “Code”, so a Hierarchy name must be specified. Measures are normally “Numbers”, here an interval could be stated. Mandatory if “ConceptDataType” is “Code”",
    ("datasetdef", "concept_description"): "Description of the dataset (first appearance of the dataset) or Concept (following rows).",
    ("datasetdef", "attributes"): "A list of freely definable attributes attached to the dataset or concept.",
    # Parameters
    ("parameters", "name"): "Name of the parameter. Must be unique across the case study.",
    ("parameters", "type"): "Type of the parameter.",
    ("parameters", "domain"): "Set of potential values for the parameter. Mandatory for “Code”, an optional range for “Number”",
    ("parameters", "value"): "Either a literal value or a conditional list of expressions which can have other parameters (declared previously). An empty value is permitted because “Solver” command can define -or redefine- parameters.",
    ("parameters", "group"): "Name of the group of parameters to which the current parameter is a member of, just for readability.",
    ("parameters", "description"): "A description of the meaning and intent of the parameter.",
    ("parameters", "attributes"): "A list of freely definable attributes attached to the parameter. Columns with a header starting in “@” are considered attribute columns that are appended to the list of attributes.",
    # InterfaceTypes
    ("interface_types", "interface_type_hierarchy"): "Name of the hierarchy to which the InterfaceType is a member.",
    ("interface_types", "interface_type"): "Name of the InterfaceType. Unique among all Interface Types.",
    ("interface_types", "sphere"): "The default value of sphere, either “Technosphere” or “Biosphere”, for Interfaces of this type (can be overridden by each Interface when it is declared). Indicates the type of process originating the input flow. Or the destination for output flows.",
    ("interface_types", "roegen_type"): "The default Either “Flow” or “Fund”, indicates the type of factor for the interface in accordance with Roegen theory.",
    ("interface_types", "parent_interface_type"): "Name of the parent InterfaceType, if any.",
    ("interface_types", "formula"): "A formula combining arithmetically InterfaceTypes in different branches of the same hierarchy.",
    ("interface_types", "description"): "Description of the InterfaceType.",
    ("interface_types", "unit"): "The default unit for quantities attached to the InterfaceType. A very large list of units is recognized.",
    ("interface_types", "opposite_processor_type"): "Which of the case study contexts, “Local”, “External” or “Environment”, does the Processor opposite to where Interfaces of the InterfaceType pertain to.",
    ("interface_types", "attributes"): "A list of freely definable attributes attached to an InterfaceType. Columns with a header starting in “@” are considered attribute columns that are appended to the list of attributes.",
    # Processors
    ("processors", "processor_group"): "A group to which the Processor will be a member of.",
    ("processors", "processor"): "Name of the Processor.",
    ("processors", "parent_processor"): "Name of the parent Processor regarding “part-of” relation. It is a quick way of specifying a hierarchy of functional composition. Another way is through the command “Relationships”.",
    ("processors", "subsystem_type"): "The main context of the Processor. Default value 'Local'.",
    ("processors", "processor_system"): "The system to which the processor is member of. It is allowed to connect processors from multiple systems (the initial intent is to label a system as a country, like “ES”, “NL”, “IT”). Default value: 'default'",
    ("processors", "functional_or_structural"): "One of two options must be specified: Functional or Structural. “Functional” means the Processor is a function, which is the goal in the abstract, opposed to “Structure” which is a Processor representing a materialized goal. Structural Processors should be part-of functional Processors (but structural Processors can have other structural Processors inside, and functional Processors can have functional Processors inside).",
    ("processors", "instance_or_archetype"): "One of two options must be specified: Instance or Archetype. “Instance” means the Processor will be incorporated in the accounting process, while “Archetype” is used because the Processor will be just a template for other Processors that will be copies of it (for instance intensive processors that will be copied and scaled up).",
    ("processors", "stock"): "Inform that the Processor is a stock (“Yes”) or a producer/consumer (“No”)",
    ("processors", "description"): "Description of the Processor.",
    ("processors", "geolocation_ref"): "The ID of a “geographic dataset” reference.",
    ("processors", "geolocation_code"): "Either a Code in a Hierarchy having a geographic object attached to it, or the code of an object inside the spatial dataset specified in “GeolocationRef”.",
    ("processors", "attributes"): "A list of freely definable attributes attached to a Processor. Columns with a header starting in “@” are considered attribute columns that are appended to the list of attributes.",
    # Interfaces
    ("interfaces_and_qq", "processor"): "Processor to which the Interface is added. Or the quantity, if the Interface has been already added.",
    ("interfaces_and_qq", "interface_type"): "Name of the Interface Type to be used.",
    ("interfaces_and_qq", "interface"): "Name of the Interface in case it is needed because by default the InterfaceType name is used for the Interface (if two Interfaces with the same InterfaceType are needed in the same Processor, they must have different names).",
    ("interfaces_and_qq", "sphere"): "The sphere, either “Technosphere” or “Biosphere”, for the Interface being defined, potentially overriding the equivalent property of InterfaceType. Indicates the type of process (artificial or natural) originating the input flow. Or the destination for output flows.",
    ("interfaces_and_qq", "roegen_type"): "Either “Flow” or “Fund”, indicates the type of factor for the Interface in accordance with Roegen’s theory. Overrides the RoegenType of the InterfaceType if it was specified.",
    ("interfaces_and_qq", "orientation"): "Orientation, “Input” or “Output”, of the Interface. Overrides the Orientation of the InterfaceType if it was specified.",
    ("interfaces_and_qq", "opposite_processor_type"): "Used to point which of the case study contexts do Processors opposite to the Interface pertain to.",
    ("interfaces_and_qq", "geolocation_ref"): "Reference to information on where the Interface is located.",
    ("interfaces_and_qq", "geolocation_code"): "Either a Code in a Hierarchy having a geographic object attached to it, or the code of an object inside the spatial dataset specified in “GeolocationRef”.",
    ("interfaces_and_qq", "interface_attributes"): "A list of freely definable attributes attached to an Interface. Columns with a header starting in “I@” are considered attribute columns that are appended to the list of attributes of the Interface. Quantities have a separate list of attributes, see QuantityAttributes.",
    ("interfaces_and_qq", "value"): "An arithmetic-boolean expression with parameters.",
    ("interfaces_and_qq", "unit"): "Unit for Value. A very large list of units is recognized. Overrides the Unit of the InterfaceType, if it was specified.",
    ("interfaces_and_qq", "relative_to"): "If used, it would be the name of an Interface defining the unit to scale the currently stated Value. For instance, if the Unit is “kg” and “RelativeTo” is “LU” in hectares, once “LU” is quantified, lets assume in “10 ha”, the resulting amount would be multiplied by 10, and the unit would remain to be “kg”.",
    ("interfaces_and_qq", "uncertainty"): "Statistical characterization of the uncertainty of Value. This field should be elaborated together with solvers capable of exploiting the uncertainty information: sensitivity analysis, observability analysis (recognize the inherent strength-weakness of assessment including this measure).",
    ("interfaces_and_qq", "assessment"): "Assessment according to NUSAP. This should be elaborated together with solvers capable of symbolically analyzing categorizations of assessment.",
    ("interfaces_and_qq", "pedigree_matrix"): "Reference to a previously declared PedigreeMatrix (“NUSAP.PM” command).",
    ("interfaces_and_qq", "pedigree"): "Codes in the PedigreeMatrix encoding qualities of the diagnostic quantity.",
    ("interfaces_and_qq", "time"): "To which time period the quantity corresponds to.",
    ("interfaces_and_qq", "qq_source"): "Reference to a source defining how (provenance) or from (bibliography) the quantity was obtained.",
    ("interfaces_and_qq", "number_attributes"): "A list of freely definable attributes attached to a Quantity. Columns with a header starting in “N@” are considered attribute columns that are appended to the list of attributes of the Quantity",
    ("interfaces_and_qq", "comments"): "Free comments attached to the quantity.",
    # Relationships
    ("relationships", "source_processor"): "An expression resolving to a set of processors that will be in the origin part of the relations (one or more) to be specified. If “Origin” field is not specified, this field is mandatory.",
    ("relationships", "source_interface"): "The name of an Interface present in all the Processors in the set of origin Processors. When connecting Processors it can be empty. Also, when connecting Interfaces, if the destination Interface is specified, an equally named Interface is assumed in the origin Processor.",
    ("relationships", "target_processor"): "An expression resolving to a set of processors that will be in the destination part of the relations (one or more) to be specified. If “Destination” field is not specified, this field is mandatory.",
    ("relationships", "target_interface"): "The name of an Interface present in all the Processors in the set of destination Processors. When connecting Processors it can be empty. Also, when connecting Interfaces, if the origin Interface is specified, an equally named Interface is assumed in the destination Processor.",
    ("relationships", "back_interface"): "The name of an Interface in the origin processor, used if a change of scale is requested, allowing to account back to the origin processor the amount. It is intended for expressing the accounting of money obtained in exchange for some good.",
    ("relationships", "relation_type"): "Please, refer to MAGIC - ""DMP Annex 3"" (text too long to show here)",
    ("relationships", "flow_weight"): "Proportion of the magnitude of the Origin Interface that goes into the Destination Interface.",
    ("relationships", "source_cardinality"): "Only for “Associate” relations",
    ("relationships", "target_cardinality"): "Only for “Associate” relations",
    ("relationships", "attributes"): "A list of freely definable attributes attached to a Relationship. Columns with a header starting in “@” are considered attribute columns that are appended to the list of attributes.",
    # ProcessorScalings
    ("processor_scalings", "invoking_processor"): "Parent processor for hierarchical scales, origin sibling processor for sequential scales.",
    ("processor_scalings", "requested_processor"): "Child processor for hierarchical scales, destination processor for sequential scales. Depending on the ScalingType, this Processor will exist previously or it will be created from a template processor (unit -or intensive- processor).",
    ("processor_scalings", "scaling_type"): "Please, refer to MAGIC - ""DMP Annex 3"" (text too long to show here)",
    ("processor_scalings", "invoking_interface"): "Interface to be used for the invoking processor.",
    ("processor_scalings", "requested_interface"): "Interface to be used for the requested processor.",
    ("processor_scalings", "scale"): "Factor by which invoking interface is multiplied to define the requested interface.",
    ("processor_scalings", "new_processor_name"): "If a new processor is created (“CloneAndScale” is used) the name of the processor is in this field. If not, the name of the requested processor is used.",
    ("processor_scalings", "processor_group"): "If a new processor is created ('CloneAndScale' is used) the group to which the processor is added",
    ("processor_scalings", "parent_processor"): "If a new processor is created ('CloneAndScale' is used) the parent of the new processor",
    ("processor_scalings", "subsystem_type"): "If a new processor is created ('CloneAndScale' is used) the context of the new Processor",
    ("processor_scalings", "processor_system"): "If a new processor is created ('CloneAndScale' is used), the system to which the processor is member of",
    ("processor_scalings", "description"): "If a new processor is created ('CloneAndScale' is used), the description attached to the new processor",
    ("processor_scalings", "geolocation_ref"): "If a new processor is created ('CloneAndScale' is used), ID of a geolocation reference for this new processor",
    ("processor_scalings", "geolocation_code"): "If a new processor is created ('CloneAndScale' is used), Code in a hierarchy or geographic reference",
    ("processor_scalings", "attributes"): "If a new processor is created ('CloneAndScale' is used), attributes for the new processor",
    # ScaleChangeMap
    ("scale_conversion_v2", "source_hierarchy"): "Name of the origin InterfaceTypes Hierarchy. It is optional if the origin InterfaceType name is unique among InterfaceTypes. Otherwise, it must be specified.",
    ("scale_conversion_v2", "source_interface_type"): "Name of the InterfaceType in the origin InterfaceTypes Hierarchy.",
    ("scale_conversion_v2", "target_hierarchy"): "Name of the destination InterfaceTypes Hierarchy. It is optional if the destination InterfaceType name is unique among InterfaceTypes. Otherwise, it must be specified.",
    ("scale_conversion_v2", "target_interface_type"): "Name of a code in the destination InterfaceTypes Hierarchy.",
    ("scale_conversion_v2", "source_context"): "Specification of which Processors are to be used to match the Origin part of the context.",
    ("scale_conversion_v2", "target_context"): "Specification of which Processors are to be used to match the Destination part of the context.",
    ("scale_conversion_v2", "scale"): "Linear scale transform from origin to destination. Both origin and destination can appear multiple times, so the multiplicity potentially can be many to many.",
    ("scale_conversion_v2", "source_unit"): "Specification of unit for which the scale factor is valid. If quantities for Interfaces are expressed in other physically equivalent unit, automatic unit conversion will be performed before the scaling.",
    ("scale_conversion_v2", "target_unit"): "Specification of unit (destination) for which the scale factor is valid. If quantities for Interfaces are expressed in other physically equivalent unit, automatic unit conversion will be performed before the scaling.",
    # ScalarBenchmarks
    ("scalar_indicator_benchmarks", "benchmark_group"):
    "To which of the set of predefined groups this benchmark is member of",
    ("scalar_indicator_benchmarks", "stakeholders"):
    "A list of names that can be used to frame which stakeholders consider/use this benchmark.",
    ("scalar_indicator_benchmarks", "benchmark"): "Name of the benchmark",
    ("scalar_indicator_benchmarks", "range"): "Numeric interval capturing one of the categories of the benchmark.",
    ("scalar_indicator_benchmarks", "unit"): "The unit under which the range is expressed.",
    ("scalar_indicator_benchmarks", "category"):
    "A relatively formal name that can be used later to elaborate visualizations (color, icon, ...)",
    ("scalar_indicator_benchmarks", "label"): "A short string sketching the level or category inside the benchmark.",
    ("scalar_indicator_benchmarks", "description"): "Description of the category or the benchmark.",
    # ScalarIndicators
    ("scalar_indicators", "indicator_name"): "Name for the indicator, will appear in result matrices",
    ("scalar_indicators", "local"):
    "'Yes' if the formula for the indicator uses values attached to a single processor. 'No' if it is a formula "
    "referring to values in the whole model. Only 'Yes' allowed currently.",
    ("scalar_indicators", "formula"):
    "Arithmetic expression to calculate the indicator. If it is local, it can refer to interface names and also to "
    "scenario parameters",
    ("scalar_indicators", "benchmarks"): "Name of benchmarks under which the indicator is framed.",
    ("scalar_indicators", "description"): "Meaning of the indicator.",
    # MatrixIndicators
    ("matrix_indicators", "indicator_name"): "Name of the matrix indicator.",
    ("matrix_indicators", "scope"): "Used to enable quantifying openness",
    ("matrix_indicators", "processors_selector"): "Used to obtain a set of processors for the rows of the matrix.",
    ("matrix_indicators", "interfaces_selector"): "Interfaces to place in the columns of the matrix.",
    ("matrix_indicators", "indicators_selector"): "Local scalar indicators to place in the columns of the matrix.",
    ("matrix_indicators", "attributes_selector"): "Attributes (from processor) to place in the columns of the matrix.",
    ("matrix_indicators", "description"): "Intent of the indicator.",
    # ProblemStatement
    ("problem_statement", "scenario_name"): "Name of the scenario.",
    ("problem_statement", "parameter"): "Name of the parameter that is being modified.",
    ("problem_statement", "parameter_value"): "",
    ("problem_statement", "description"): "Explanation of the value or the intent of the scenario.",
}
