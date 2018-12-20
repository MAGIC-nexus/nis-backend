import re

from backend import CommandField
from backend.command_generators.parser_field_parsers import simple_ident, unquoted_string, alphanums_string, \
    hierarchy_expression_v2, key_value_list, key_value, parameter_value, expression_with_parameters, \
    time_expression, boolean, indicator_expression, code_string, simple_h_name, domain_definition, context_query, \
    unit_name, geo_value, url_parser, processor_names, date, value, list_simple_ident, reference
from backend.models.musiasem_concepts import Processor

data_types = ["Number", "Boolean", "URL", "UUID", "Datetime", "String", "UnitName", "Code", "Geo"]
concept_types = ["Dimension", "Measure", "Attribute"]
parameter_types = ["Number", "Code", "Boolean"]
element_types = ["Parameter", "Processor", "InterfaceType", "Interface"]
spheres = ["Biosphere", "Technosphere"]
roegen_types = ["Flow", "Fund"]
orientations = ["Input", "Output"]
no_yes = ["No", "Yes"]
yes_no = ["Yes", "No"]
processor_types = ["Local", "Environment", "External", "ExternalEnvironment"]
functional_or_structural = ["Functional", "Structural"]
instance_or_archetype = ["Instance", "Archetype"]
copy_interfaces_mode = ["No", "FromParent", "FromChildren", "Bidirectional"]
source_cardinalities = ["One", "Zero", "ZeroOrOne", "ZeroOrMore", "OneOrMore"]
target_cardinalities = source_cardinalities
relation_types = [# Relationships between Processors
                  "is_a", "IsA",  # "Left" gets a copy of ALL "Right" interface types
                  "as_a", "AsA",  # Left must already have ALL interfaces from Right. Similar to "part-of" in the sense that ALL Right interfaces are connected from Left to Right
                  "part_of", "|", "PartOf",  # The same thing. Left is inside Right. No assumptions on flows between child and parent.
                  "aggregate", "aggregation",
                  "compose",
                  "associate", "association",
                  # Relationships between interfaces
                  "flow", "scale", ">", "<"
                  ]
processor_scaling_types = ["CloneAndScale", "Scale", "CloneScaled"]
agent_types = ["Person", "Software", "Organization"]
geographic_resource_types = ["dataset"]
geographic_topic_categories = ["Farming", "Biota", "Boundaries", "Climatology", "Meteorology", "Atmosphere", "Economy", "Elevation", "Environment", "GeoscientificInformation", "Health", "Imagery", "BaseMaps", "EarthCover", "Intelligence", "Military", "InlandWaters", "Location", "Oceans", "Planning", "Cadastre", "Society", "Structure", "Transportation", "Utilities", "Communication"]
bib_entry_types = ["article", "book", "booklet", "conference", "inbook", "incollection", "inproceedings",
                   "manual", "mastersthesis", "misc", "phdtesis", "proceedings", "techreport", "unpublished"]
bib_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
command_names = ["Structure", "Scale", "Parameters", "Indicators", "Hierarchies", "HierarchiesMapping",
                 "AttributeTypes", "DatasetDef", "DatasetData", "DatasetQry", "InterfaceTypes", "BareProcessors",
                 "Interfaces", "Relationships", "ProcessorScalings", "ScaleChangeMap",
                 "RefBibliographic", "RefGeographic", "RefProvenance", "ProblemStatement"]

attributeRegex = "@.+"

commands = {
    "CatHierarchies":
    [CommandField(allowed_names=["Source"], name="source", mandatory=False, allowed_values=None, parser=simple_h_name),
     CommandField(allowed_names=["HierarchyGroup"], name="hierarchy_group", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Hierarchy", "HierarchyName"], name="hierarchy_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Level", "LevelCode"], name="level", mandatory=False, allowed_values=None, parser=alphanums_string),
     CommandField(allowed_names=["ReferredHierarchy"], name="referred_hierarchy", mandatory=False, allowed_values=None, parser=simple_h_name),
     CommandField(allowed_names=["Code"], name="code", mandatory=False, allowed_values=None, parser=code_string),
     # NOTE: Removed because parent code must be already a member of the hierarchy being defined
     # CommandField(allowed_names=["ReferredHierarchyParent"], name="referred_hierarchy_parent", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["ParentCode"], name="parent_code", mandatory=False, allowed_values=None, parser=code_string),
     CommandField(allowed_names=["Label"], name="label", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Description"], name="description", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Expression", "Formula"], name="expression", mandatory=False, allowed_values=None, parser=hierarchy_expression_v2),
     CommandField(allowed_names=["GeolocationRef"], name="geolocation_ref", mandatory=False, allowed_values=None, parser=reference),
     CommandField(allowed_names=["GeolocationCode"], name="geolocation_code", mandatory=False, allowed_values=None, parser=code_string),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=[attributeRegex], name="attributes", mandatory=False, many_appearances=True, parser=value)
     ],
    "CatHierarchiesMapping":
    [CommandField(allowed_names=["OriginDataset"], name="source_dataset", mandatory=False, allowed_values=None, parser=simple_h_name),
     CommandField(allowed_names=["OriginHierarchy"], name="source_hierarchy", mandatory=True, allowed_values=None, parser=simple_h_name),
     CommandField(allowed_names=["OriginCode"], name="source_code", mandatory=True, allowed_values=None, parser=code_string),
     CommandField(allowed_names=["DestinationHierarchy"], name="destination_hierarchy", mandatory=True, allowed_values=None, parser=simple_h_name),
     CommandField(allowed_names=["DestinationCode"], name="destination_code", mandatory=True, allowed_values=None, parser=code_string),
     CommandField(allowed_names=["Weight"], name="weight", mandatory=True, allowed_values=None, parser=expression_with_parameters),
     ],
    "AttributeTypes":
    [CommandField(allowed_names=["AttributeType", "AttributeTypeName"], name="attribute_type_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Type"], name="data_type", mandatory=True, allowed_values=data_types, parser=simple_ident),
     CommandField(allowed_names=["ElementTypes"], name="element_types", mandatory=False, allowed_values=element_types, parser=list_simple_ident),
     CommandField(allowed_names=["Domain"], name="domain", mandatory=False, allowed_values=None, parser=domain_definition)  # "domain_definition" for Category and NUmber. Boolean is only True or False. Other data types cannot be easily constrained (URL, UUID, Datetime, Geo, String)
     ],
    "DatasetDef":
    [CommandField(allowed_names=["Dataset", "DatasetName"], name="dataset_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DatasetDataLocation"], name="dataset_data_location", mandatory=True, allowed_values=None, parser=url_parser),
     CommandField(allowed_names=["ConceptType"], name="concept_type", mandatory=True, allowed_values=concept_types, parser=simple_ident),
     CommandField(allowed_names=["Concept", "ConceptName"], name="concept_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DataType", "ConceptDataType"], name="concept_data_type", mandatory=True, allowed_values=data_types, parser=simple_ident),
     CommandField(allowed_names=["Domain", "ConceptDomain"], name="concept_domain", mandatory=False, allowed_values=None, parser=domain_definition),
     CommandField(allowed_names=["Description", "ConceptDescription"], name="concept_description", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     ],
    # "DatasetData" needs a specialized parser
    "AttributeSets":
    [CommandField(allowed_names=["AttributeSetName"], name="attribute_set_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=[attributeRegex], name="attributes", mandatory=False, many_appearances=True, parser=value)
     ],
    "Parameters":
    [CommandField(allowed_names=["Parameter", "ParameterName"], name="name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Type"], name="type", mandatory=True, allowed_values=parameter_types, parser=simple_ident),
     CommandField(allowed_names=["Domain"], name="domain", mandatory=False, allowed_values=None, parser=domain_definition),
     CommandField(allowed_names=["Value"], name="value", mandatory=False, allowed_values=None, parser=expression_with_parameters),
     CommandField(allowed_names=["Group"], name="group", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Description"], name="description", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=[attributeRegex], name="attributes", mandatory=False, many_appearances=True, parser=value)
     ],
    # "DatasetQry" needs a specialized parser
    "InterfaceTypes":
    [CommandField(allowed_names=["InterfaceTypeHierarchy"], name="interface_type_hierarchy", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["InterfaceType"], name="interface_type", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Sphere"], name="sphere", mandatory=True, allowed_values=spheres, parser=simple_ident),
     CommandField(allowed_names=["RoegenType"], name="roegen_type", mandatory=True, allowed_values=roegen_types, parser=simple_ident),
     CommandField(allowed_names=["ParentInterfaceType"], name="parent_interface_type", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Formula", "Expression"], name="formula", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Description"], name="description", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Unit"], name="unit", mandatory=False, allowed_values=None, parser=unit_name),
     CommandField(allowed_names=["Orientation"], name="orientation", mandatory=False, allowed_values=orientations, parser=simple_ident),
     CommandField(allowed_names=["OppositeProcessorType"], name="opposite_processor_type", mandatory=False, allowed_values=processor_types, parser=simple_ident),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=[attributeRegex], name="attributes", mandatory=False, many_appearances=True, parser=value)
     ],
    "Processors":
    [CommandField(allowed_names=["ProcessorGroup"], name="processor_group", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Processor"], name="processor", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["ParentProcessor"], name="parent_processor", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["CopyInterfaces"], name="copy_interfaces_mode", mandatory=False, allowed_values=copy_interfaces_mode, parser=simple_ident),
     CommandField(allowed_names=["CloneProcessor"], name="clone_processor", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["ProcessorContextType", "ProcessorType"], name="processor_type", mandatory=False, allowed_values=processor_types, parser=simple_ident, attribute_of=Processor),
     CommandField(allowed_names=["FunctionalOrStructural"], name="functional_or_structural", mandatory=False, allowed_values=functional_or_structural, parser=simple_ident, attribute_of=Processor),
     CommandField(allowed_names=["InstanceOrArchetype"], name="instance_or_archetype", mandatory=False, allowed_values=instance_or_archetype, parser=simple_ident, attribute_of=Processor),
     CommandField(allowed_names=["Stock"], name="stock", mandatory=False, allowed_values=no_yes, parser=simple_ident, attribute_of=Processor),
     CommandField(allowed_names=["Alias", "SpecificName"], name="alias", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Description"], name="description", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["GeolocationRef"], name="geolocation_ref", mandatory=False, allowed_values=None, parser=reference),
     CommandField(allowed_names=["GeolocationCode"], name="geolocation_code", mandatory=False, allowed_values=None, parser=code_string),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list, attribute_of=Processor),
     CommandField(allowed_names=[attributeRegex], name="attributes", mandatory=False, many_appearances=True, parser=value),
     ],
    "Interfaces":
    [CommandField(allowed_names=["Alias", "SpecificName"], name="alias", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["InterfaceType"], name="interface_type", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Interface"], name="interface", mandatory=False, allowed_values=None, parser=simple_ident),  # Processor:InterfaceType
     # TODO
     #CommandField(allowed_names=["Processor"], name="processor", mandatory=True, allowed_values=None, parser=processor_name),
     CommandField(allowed_names=["Processor"], name="processor", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Sphere"], name="sphere", mandatory=False, allowed_values=spheres, parser=simple_ident),
     CommandField(allowed_names=["RoegenType"], name="roegen_type", mandatory=False, allowed_values=roegen_types, parser=simple_ident),
     CommandField(allowed_names=["Orientation"], name="orientation", mandatory=False, allowed_values=orientations, parser=simple_ident),
     CommandField(allowed_names=["OppositeProcessorType"], name="opposite_processor_type", mandatory=False, allowed_values=processor_types, parser=simple_ident),
     CommandField(allowed_names=["GeolocationRef"], name="geolocation_ref", mandatory=False, allowed_values=None, parser=reference),
     CommandField(allowed_names=["GeolocationCode"], name="geolocation_code", mandatory=False, allowed_values=None, parser=code_string),
     CommandField(allowed_names=["InterfaceAttributes"], name="interface_attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["I"+attributeRegex], name="interface_attributes", mandatory=False, many_appearances=True, parser=value),

     # Qualified Quantification
     CommandField(allowed_names=["Value"], name="value", mandatory=False, allowed_values=None, parser=expression_with_parameters),
     # TODO
     #CommandField(allowed_names=["Unit"], name="unit", mandatory=False, allowed_values=None, parser=unit_name),
     CommandField(allowed_names=["Unit"], name="unit", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Uncertainty"], name="uncertainty", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Assessment"], name="assessment", mandatory=False, allowed_values=None, parser=unquoted_string),
     # TODO
     #CommandField(allowed_names=["PedigreeMatrix"], name="pedigree_matrix", mandatory=False, allowed_values=None, parser=reference_name),
     #CommandField(allowed_names=["Pedigree"], name="pedigree", mandatory=False, allowed_values=None, parser=pedigree_code),
     #CommandField(allowed_names=["RelativeTo"], name="relative_to", mandatory=False, allowed_values=None, parser=simple_ident_plus_unit_name),
     CommandField(allowed_names=["PedigreeMatrix"], name="pedigree_matrix", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Pedigree"], name="pedigree", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["RelativeTo"], name="relative_to", mandatory=False, allowed_values=None, parser=unquoted_string),
     CommandField(allowed_names=["Time"], name="time", mandatory=False, allowed_values=None, parser=time_expression),
     CommandField(allowed_names=["Source"], name="qq_source", mandatory=False, allowed_values=None, parser=reference),
     CommandField(allowed_names=["NumberAttributes"], name="number_attributes", mandatory=False, allowed_values=None, parser=key_value_list),
     CommandField(allowed_names=["N"+attributeRegex], name="number_attributes", mandatory=False, many_appearances=True, parser=key_value),
     CommandField(allowed_names=["Comments"], name="comments", mandatory=False, allowed_values=None, parser=unquoted_string),
     ],
    "Relationships":
    [CommandField(allowed_names=["OriginProcessors", "OriginProcessor"], name="source_processor", mandatory=False, allowed_values=None, parser=processor_names),
     CommandField(allowed_names=["OriginInterface"], name="source_interface", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DestinationProcessors", "DestinationProcessor"], name="target_processor", mandatory=False, allowed_values=None, parser=processor_names),
     CommandField(allowed_names=["DestinationInterface"], name="target_interface", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Origin"], name="source", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Destination"], name="target", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["RelationType"], name="relation_type", mandatory=True, allowed_values=relation_types, parser=unquoted_string),
     CommandField(allowed_names=["ChangeOfTypeScale"], name="change_type_scale", mandatory=False, allowed_values=None, parser=expression_with_parameters),
     CommandField(allowed_names=["Weight"], name="flow_weight", mandatory=False, allowed_values=None, parser=expression_with_parameters),
     CommandField(allowed_names=["OriginCardinality"], name="source_cardinality", mandatory=False, allowed_values=source_cardinalities, parser=simple_ident),
     CommandField(allowed_names=["DestinationCardinality"], name="target_cardinality", mandatory=False, allowed_values=target_cardinalities, parser=simple_ident),
     CommandField(allowed_names=["Attributes"], name="attributes", mandatory=False, allowed_values=None, parser=key_value_list)
     ],
    "ProcessorScalings":
    [CommandField(allowed_names=["InvokingProcessor"], name="invoking_processor", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["RequestedProcessor"], name="requested_processor", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["ScalingType"], name="scaling_type", mandatory=True, allowed_values=processor_scaling_types, parser=simple_ident),
     CommandField(allowed_names=["InvokingInterface"], name="invoking_interface", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["RequestedInterface"], name="requested_interface", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Scale"], name="scale", mandatory=True, allowed_values=None, parser=expression_with_parameters),
     # TODO
     #CommandField(allowed_names=["UpscaleParentContext"], name="upscale_parent_context", mandatory=False, allowed_values=None, parser=upscale_context),
     #CommandField(allowed_names=["UpscaleChildContext"], name="upscale_child_context", mandatory=False, allowed_values=None, parser=upscale_context),
     ],
    "ScaleChangers":
    [CommandField(allowed_names=["OriginHierarchy"], name="source_hierarchy", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["OriginInterfaceType"], name="source_interface_type", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DestinationHierarchy"], name="target_hierarchy", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["DestinationInterfaceType"], name="target_interface_type", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["OriginContext"], name="source_context", mandatory=False, allowed_values=None, parser=processor_names),
     CommandField(allowed_names=["DestinationContext"], name="target_context", mandatory=False, allowed_values=None, parser=processor_names),
     CommandField(allowed_names=["Scale"], name="scale", mandatory=False, allowed_values=None, parser=expression_with_parameters),
     #CommandField(allowed_names=["OriginUnit"], name="source_unit", mandatory=False, allowed_values=None, parser=unit_name),
     #CommandField(allowed_names=["DestinationUnit"], name="target_unit", mandatory=False, allowed_values=None, parser=unit_name),
     CommandField(allowed_names=["OriginUnit"], name="source_unit", mandatory=False, allowed_values=None, parser=unit_name),
     CommandField(allowed_names=["DestinationUnit"], name="target_unit", mandatory=False, allowed_values=None, parser=unit_name)
     ],
    # "SharedElements":
    # [
    #  ],
    # "ReusedElements":
    # [
    #  ],
    "ImportCommands":
    [
        CommandField(allowed_names=["Workbook", "WorkbookLocation"], name="workbook_name", mandatory=False, allowed_values=None, parser=url_parser),
        CommandField(allowed_names=["Worksheets"], name="worksheets", mandatory=False, allowed_values=None, parser=unquoted_string)
    ],
    "TableOfContents":
    [
        CommandField(allowed_names=["Worksheet", "WorksheetName"], name="worksheet", mandatory=True, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Command"], name="command", mandatory=True, allowed_values=command_names, parser=simple_ident),
        CommandField(allowed_names=["Comment", "Description"], name="comment", mandatory=False, allowed_values=None, parser=unquoted_string),
    ],
    "RefProvenance":
    [   # Reduced, from W3C Provenance Recommendation
        CommandField(allowed_names=["RefID", "Reference"], name="ref_id", mandatory=True, allowed_values=None, parser=simple_ident),
        CommandField(allowed_names=["AgentType"], name="agent_type", mandatory=True, allowed_values=agent_types, parser=simple_ident),
        CommandField(allowed_names=["Agent"], name="agent", mandatory=True, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Activities"], name="activities", mandatory=True, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Entities"], name="entities", mandatory=False, allowed_values=None, parser=unquoted_string),
    ],
    "RefGeographic":
    [   # A subset of fields from INSPIRE regulation for metadata: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32008R1205&from=EN
        # Fields useful to elaborate graphical displays. Augment in the future as demanded
        CommandField(allowed_names=["RefID", "Reference"], name="ref_id", mandatory=True, allowed_values=None, parser=simple_ident),
        CommandField(allowed_names=["Title"], name="title", mandatory=True, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Abstract"], name="abstract", mandatory=False, allowed_values=None, parser=unquoted_string),  # Syntax??
        CommandField(allowed_names=["Type"], name="type", mandatory=False, allowed_values=geographic_resource_types, parser=unquoted_string),  # Part D.1. JUST "Dataset"
        CommandField(allowed_names=["ResourceLocator", "DataLocation"], name="data_location", mandatory=False, allowed_values=None, parser=url_parser),
        CommandField(allowed_names=["TopicCategory"], name="topic_category", mandatory=False, allowed_values=geographic_topic_categories, parser=unquoted_string),  # Part D.2
        CommandField(allowed_names=["BoundingBox"], name="bounding_box", mandatory=False, allowed_values=None, parser=unquoted_string),  # Syntax??
        CommandField(allowed_names=["TemporalExtent", "Date"], name="temporal_extent", mandatory=False, allowed_values=None, parser=unquoted_string),  # Syntax??
        CommandField(allowed_names=["PointOfContact"], name="metadata_point_of_contact", mandatory=False, allowed_values=None, parser=unquoted_string),
    ],
    "RefBibliographic":
    [   # From BibTex. Mandatory fields depending on EntryType, at "https://en.wikipedia.org/wiki/BibTeX" (or search: "Bibtex entry field types")
        CommandField(allowed_names=["RefID", "Reference"], name="ref_id", mandatory=True, allowed_values=None, parser=simple_ident),
        CommandField(allowed_names=["EntryType"], name="entry_type", mandatory=True, allowed_values=bib_entry_types, parser=unquoted_string),
        CommandField(allowed_names=["Address"], name="address", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Annote"], name="annote", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Author"], name="author", mandatory="entry_type not in ('booklet', 'manual', 'misc', 'proceedings')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["BookTitle"], name="booktitle", mandatory="entry_type in ('incollection', 'inproceedings')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Chapter"], name="chapter", mandatory="entry_type in ('inbook')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["CrossRef"], name="crossref", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Edition"], name="edition", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Editor"], name="editor", mandatory="entry_type in ('book', 'inbook')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["HowPublished"], name="how_published", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Institution"], name="institution", mandatory="entry_type in ('techreport')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Journal"], name="journal", mandatory="entry_type in ('article')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Key"], name="key", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Month"], name="month", mandatory=False, allowed_values=bib_months, parser=simple_ident),
        CommandField(allowed_names=["Note"], name="note", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Number"], name="number", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Organization"], name="organization", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Pages"], name="pages", mandatory="entry_type in ('inbook')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Publisher"], name="publisher", mandatory="entry_type in ('book', 'inbook', 'incollection')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["School"], name="school", mandatory="entry_type in ('mastersthesis', 'phdtesis')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Series"], name="series", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Title"], name="title", mandatory="entry_type not in ('misc')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Type"], name="type", mandatory=False, allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["URL"], name="url", mandatory=False, allowed_values=None, parser=url_parser),
        CommandField(allowed_names=["Volume"], name="volume", mandatory="entry_type in ('article')", allowed_values=None, parser=unquoted_string),
        CommandField(allowed_names=["Year"], name="year", mandatory="entry_type in ('article', 'book', 'inbook', 'incollection', 'inproceedings', 'mastersthesis', 'phdthesis', 'proceedings', 'techreport')", allowed_values=None, parser=unquoted_string)
    ],
    "ScalarIndicators":
    [CommandField(allowed_names=["Indicator"], name="indicator_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Local"], name="local", mandatory=True, allowed_values=yes_no, parser=simple_ident),
     CommandField(allowed_names=["Formula", "Expression"], name="expression", mandatory=True, allowed_values=None, parser=indicator_expression),
     CommandField(allowed_names=["Benchmark"], name="benchmark", mandatory=False, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Description"], name="description", mandatory=False, allowed_values=None, parser=unquoted_string),
     ],
    "MatrixIndicators":
    [CommandField(allowed_names=["Indicator"], name="indicator_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Formula", "Expression"], name="expression", mandatory=True, allowed_values=None, parser=indicator_expression),
     CommandField(allowed_names=["Description"], name="description", mandatory=False, allowed_values=None, parser=unquoted_string),
     ],
    "ProblemStatement":
    [CommandField(allowed_names=["Scenario"], name="scenario_name", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Parameter"], name="parameter", mandatory=True, allowed_values=None, parser=simple_ident),
     CommandField(allowed_names=["Value"], name="parameter_value", mandatory=True, allowed_values=None, parser=expression_with_parameters),
     CommandField(allowed_names=["Description"], name="description", mandatory=False, allowed_values=None, parser=unquoted_string),
     ]
}

# command_field_names = {}
# for command in commands.values():
#     for field in command:
#         for name in field.allowed_names:
#             command_field_names[name] = field.name

command_field_names = {name: f.name for cmd in commands.values() for f in cmd for name in f.allowed_names}
print(f'command_field_names = {command_field_names}')


def compile_command_field_regexes():
    def contains_any(s, setc):
        return 1 in [c in s for c in setc]

    for cmd in commands:
        # Compile the regular expressions of column names
        flags = re.IGNORECASE
        for c in commands[cmd]:
            rep = [(r if contains_any(r, ".+") else re.escape(r))+"$" for r in c.allowed_names]
            c.regex_allowed_names = re.compile("|".join(rep), flags=flags)


# Execute the previous function
compile_command_field_regexes()
