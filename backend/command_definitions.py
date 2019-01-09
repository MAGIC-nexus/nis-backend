######################
#  LIST OF COMMANDS  #
######################
from typing import List

from backend import Command, _var_name, _hvar_name, _cplex_var
from backend.command_generators.spreadsheet_command_parsers.external_data.etl_external_dataset_spreadsheet_parse import \
    parse_etl_external_dataset_command
from backend.command_generators.spreadsheet_command_parsers.external_data.mapping_spreadsheet_parse import parse_mapping_command
from backend.command_generators.spreadsheet_command_parsers.specification.data_input_spreadsheet_parse import \
    parse_data_input_command
from backend.command_generators.spreadsheet_command_parsers.specification.hierarchy_spreadsheet_parser import \
    parse_hierarchy_command
from backend.command_generators.spreadsheet_command_parsers.specification.metadata_spreadsheet_parse import \
    parse_metadata_command
from backend.command_generators.spreadsheet_command_parsers.specification.pedigree_matrix_spreadsheet_parse import \
    parse_pedigree_matrix_command
from backend.command_generators.spreadsheet_command_parsers.specification.references_spreadsheet_parser import \
    parse_references_command
from backend.command_generators.spreadsheet_command_parsers.specification.scale_conversion_spreadsheet_parse import \
    parse_scale_conversion_command
from backend.command_generators.spreadsheet_command_parsers.specification.structure_spreadsheet_parser import \
    parse_structure_command
from backend.command_generators.spreadsheet_command_parsers.specification.upscale_spreadsheet_parse import parse_upscale_command
from backend.command_generators.spreadsheet_command_parsers_v2.dataset_data_spreadsheet_parse import parse_dataset_data_command
from backend.command_generators.spreadsheet_command_parsers_v2.dataset_qry_spreadsheet_parse import parse_dataset_qry_command

commands: List[Command] = [
    Command(name="dummy", labels=["Dummy"], is_v1=True,
            execution_class_name="backend.command_executors.specification.dummy_command.DummyCommand"),

    # Version 1 only

    Command(name="data_input", labels=["DataInput"], is_v1=True,
            execution_class_name="backend.command_executors.specification.data_input_command.DataInputCommand",
            parse_function=parse_data_input_command,
            alt_regex=r"(Processors|Proc)[ _]+" + _var_name),

    Command(name="hierarchy", labels=["Hierarchy"], is_v1=True,
            execution_class_name="backend.command_executors.specification.hierarchy_command.HierarchyCommand",
            parse_function=parse_hierarchy_command,
            alt_regex=r"(Taxonomy|Tax|Composition|Comp)[ _]([cpf])[ ]" + _var_name),

    Command(name="upscale", labels=["Upscale"], is_v1=True,
            execution_class_name="backend.command_executors.specification.upscale_command.UpscaleCommand",
            parse_function=parse_upscale_command,
            alt_regex=r"(Upscale|Up)[ _](" + _var_name + "[ _]" + _var_name + ")?"),

    Command(name="structure", labels=["Structure"], is_v1=True,
            execution_class_name="backend.command_executors.specification.structure_command.StructureCommand",
            parse_function=parse_structure_command,
            alt_regex=r"(Grammar|Structure)([ _]+" + _var_name + ")?"),

    Command(name="scale_conversion", labels=["Scale"], is_v1=True,
            execution_class_name="backend.command_executors.specification.scale_conversion_command.ScaleConversionCommand",
            parse_function=parse_scale_conversion_command,
            alt_regex=r"Scale"),

    Command(name="pedigree_matrix", labels=["Pedigree"], is_v1=True,
            execution_class_name="backend.command_executors.specification.pedigree_matrix_command.PedigreeMatrixCommand",
            parse_function=parse_pedigree_matrix_command,
            alt_regex=r"(Pedigree|Ped|NUSAP\.PM)[ _]+" + _var_name),

    Command(name="etl_dataset", labels=["Dataset"], is_v1=True,
            execution_class_name="backend.command_executors.external_data.etl_external_dataset_command.ETLExternalDatasetCommand",
            parse_function=parse_etl_external_dataset_command,
            alt_regex=r"(Dataset|DS)[ _]" + _hvar_name),

    Command(name="mapping", labels=["Mapping"], is_v1=True,
            execution_class_name="backend.command_executors.external_data.mapping_command.MappingCommand",
            parse_function=parse_mapping_command,
            alt_regex=r"^(Mapping|Map)([ _]" + _cplex_var + "[ _]" + _cplex_var + ")?"),

    # Version 1 and maybe also Version 2

    Command(name="references", labels=["References"], is_v1=True, is_v2=True,
            execution_class_name="backend.command_executors.specification.references_command.ReferencesCommand",
            parse_function=parse_references_command,
            alt_regex=r"(References|Ref)[ _]" + _var_name),

    # Version 1 and Version 2

    Command(name="metadata", labels=["Metadata"], is_v1=True, is_v2=True,
            execution_class_name="backend.command_executors.specification.metadata_command.MetadataCommand",
            parse_function=parse_metadata_command),

    # Version 2 only

    Command(name="cat_hierarchies", labels=["CatHierarchies", "CodeHierarchies", "Hierarchies", "Categories"],
            is_v2=True,
            execution_class_name="backend.command_executors.version2.hierarchy_categories_command.HierarchyCategoriesCommand"),

    Command(name="cat_hier_mapping",
            labels=["CatHierarchiesMapping", "CodeHierarchiesMapping", "HierarchiesMapping", "CategoriesMap"],
            is_v2=True,
            execution_class_name="backend.command_executors.version2.hierarchy_mapping_command.HierarchyMappingCommand"),

    Command(name="attribute_types", labels=["AttributeTypes"], is_v2=True,
            execution_class_name="backend.command_executors.version2.attribute_types_command.AttributeTypesCommand"),

    Command(name="datasetdef", labels=["DatasetDef"], is_v2=True,
            execution_class_name="backend.command_executors.version2.dataset_definition_command.DatasetDefCommand"),

    Command(name="attribute_sets", labels=["AttributeSets"], is_v2=True,
            execution_class_name="backend.command_executors.version2.attribute_sets_command.AttributeSetsCommand"),

    Command(name="parameters", labels=["Parameters", "Params"], is_v2=True,
            execution_class_name="backend.command_executors.external_data.parameters_command.ParametersCommand"),

    Command(name="interface_types", labels=["InterfaceTypes"], is_v2=True,
            execution_class_name="backend.command_executors.version2.interface_types_command.InterfaceTypesCommand"),

    Command(name="processors", labels=["BareProcessors"], is_v2=True,
            execution_class_name="backend.command_executors.version2.processors_command.ProcessorsCommand"),

    Command(name="interfaces_and_qq", labels=["Interfaces"], is_v2=True,
            execution_class_name="backend.command_executors.version2.interfaces_command.InterfacesAndQualifiedQuantitiesCommand"),

    Command(name="relationships", labels=["Relationships", "Flows"], is_v2=True,
            execution_class_name="backend.command_executors.version2.relationships_command.RelationshipsCommand"),

    Command(name="processor_scalings", labels=["ProcessorScalings"], is_v2=True,
            execution_class_name="backend.command_executors.version2.processor_scalings_command.ProcessorScalingsCommand"),

    Command(name="scale_conversion_v2", labels=["ScaleChangeMap"], is_v2=True,
            execution_class_name="backend.command_executors.version2.scale_conversion_v2_command.ScaleConversionV2Command"),

    Command(name="import_commands", labels=["ImportCommands"], is_v2=True,
            execution_class_name="backend.command_executors.version2.nested_commands_command.NestedCommandsCommand"),

    Command(name="list_of_commands", labels=["ListOfCommands"], is_v2=True,
            execution_class_name=None),

    Command(name="ref_provenance", labels=["RefProvenance"], is_v2=True,
            execution_class_name="backend.command_executors.version2.references_v2_command.ProvenanceReferencesCommand"),

    Command(name="ref_geographical", labels=["RefGeographic", "RefGeography"], is_v2=True,
            execution_class_name="backend.command_executors.version2.references_v2_command.GeographicReferencesCommand"),

    Command(name="ref_bibliographic", labels=["RefBibliographic", "RefBibliography"], is_v2=True,
            execution_class_name="backend.command_executors.version2.references_v2_command.BibliographicReferencesCommand"),

    Command(name="scalar_indicators", labels=["ScalarIndicators"], is_v2=True,
            execution_class_name=None),

    Command(name="matrix_indicators", labels=["MatrixIndicators"], is_v2=True,
            execution_class_name=None),

    Command(name="problem_statement", labels=["ProblemStatement"], is_v2=True,
            execution_class_name="backend.command_executors.version2.problem_statement_command.ProblemSolvingCommand"),

    Command(name="datasetdata", labels=["DatasetData"], is_v2=True,
            execution_class_name="backend.command_executors.version2.dataset_data_command.DatasetDataCommand",
            parse_function=parse_dataset_data_command),

    Command(name="datasetqry", labels=["DatasetQry"], is_v2=True,
            execution_class_name="backend.command_executors.version2.dataset_query_command.DatasetQryCommand",
            parse_function=parse_dataset_qry_command),

    Command(name="shared_elements", labels=["SharedElements"], is_v2=True,
            execution_class_name="backend.command_executors.version2.export_elements_commands.ExportableElementsCommand"),

    Command(name="reused_elements", labels=["ReusedElements"], is_v2=True,
            execution_class_name="backend.command_executors.version2.import_elements_commands.ImportElementsCommand"),

    Command(name="ref_pedigree_matrices", labels=["PedigreeMatrices"], is_v2=True,
            execution_class_name="backend.command_executors.version2.pedigree_matrices_command.PedigreeMatricesReferencesCommand"),

    Command(name="indicators", labels=["Indicators"], is_v2=True,
            execution_class_name="backend.command_executors.version2.indicators_command.IndicatorsCommand",
            alt_regex=r"(Indicators|KPI)([ _]" + _var_name + ")?")
]


valid_v2_command_names: List[str] = [label for cmd in commands if cmd.is_v2 for label in cmd.labels]
