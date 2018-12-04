from backend.command_executors.version2.indicators_command import IndicatorsCommand

from backend.command_executors.external_data.etl_external_dataset_command import ETLExternalDatasetCommand
from backend.command_executors.external_data.mapping_command import MappingCommand
from backend.command_executors.external_data.parameters_command import ParametersCommand
from backend.command_executors.specification.data_input_command import DataInputCommand
from backend.command_executors.specification.dummy_command import DummyCommand
from backend.command_executors.specification.hierarchy_command import HierarchyCommand
from backend.command_executors.specification.metadata_command import MetadataCommand
from backend.command_executors.specification.pedigree_matrix_command import PedigreeMatrixCommand
from backend.command_executors.specification.references_command import ReferencesCommand
from backend.command_executors.specification.scale_conversion_command import ScaleConversionCommand
from backend.command_executors.specification.structure_command import StructureCommand
from backend.command_executors.specification.upscale_command import UpscaleCommand

from backend.command_executors.version2.attribute_sets_command import AttributeSetsCommand
from backend.command_executors.version2.attribute_types_command import AttributeTypesCommand
from backend.command_executors.version2.dataset_data_command import DatasetDataCommand
from backend.command_executors.version2.dataset_definition_command import DatasetDefCommand
from backend.command_executors.version2.dataset_query_command import DatasetQryCommand
from backend.command_executors.version2.export_elements_commands import ExportableElementsCommand
from backend.command_executors.version2.hierarchy_categories_command import HierarchyCategoriesCommand
from backend.command_executors.version2.hierarchy_mapping_command import HierarchyMappingCommand
from backend.command_executors.version2.import_elements_commands import ImportElementsCommand
from backend.command_executors.version2.nested_commands_command import NestedCommandsCommand
from backend.command_executors.version2.processor_scalings_command import ProcessorScalingsCommand
from backend.command_executors.version2.interface_types_command import InterfaceTypesCommand
from backend.command_executors.version2.interfaces_command import InterfacesAndQualifiedQuantitiesCommand
from backend.command_executors.version2.pedigree_matrices_command import PedigreeMatricesReferencesCommand
from backend.command_executors.version2.problem_statement_command import ProblemSolvingCommand
from backend.command_executors.version2.processors_command import ProcessorsCommand
from backend.command_executors.version2.references_v2_command import BibliographicReferencesCommand, \
    GeographicReferencesCommand, ProvenanceReferencesCommand
from backend.command_executors.version2.relationships_command import RelationshipsCommand
from backend.command_executors.version2.scale_conversion_v2_command import ScaleConversionV2Command


def create_command(cmd_type, name, json_input, source_block=None):
    """
    Factory creating and initializing a command from its type, optional name and parameters

    :param cmd_type: String describing the type of command, as found in the interchange JSON format
    :param name: An optional name (label) for the command
    :param json_input: Parameters specific of the command type (each command knows how to interpret,
                       validate and integrate them)
    :param source_block: String defining the name of the origin block (in a spreadsheet is the worksheet name, but other block types could appear)
    :return: The instance of the command and the issues creating it
    :raise
    """
    cmds = {"dummy":                 DummyCommand,  # Simple assignation of a string to a variable. Useful to test things
            "metadata":              MetadataCommand,
            "mapping":               MappingCommand,
            "data_input":            DataInputCommand,
            "structure":             StructureCommand,
            "upscale":               UpscaleCommand,
            "hierarchy":             HierarchyCommand,
            "etl_dataset":           ETLExternalDatasetCommand,
            "references":            ReferencesCommand,
            "pedigree_matrix":       PedigreeMatrixCommand,
            "scale_conversion":      ScaleConversionCommand,
            # V2 commands
            "cat_hierarchies":       HierarchyCategoriesCommand,
            "cat_hier_mapping":      HierarchyMappingCommand,
            "attribute_types":       AttributeTypesCommand,
            "datasetdef":            DatasetDefCommand,
            "datasetdata":           DatasetDataCommand,
            "attribute_sets":        AttributeSetsCommand,
            "parameters":            ParametersCommand,
            "datasetqry":            DatasetQryCommand,
            "interface_types":       InterfaceTypesCommand,
            "processors":            ProcessorsCommand,
            "interfaces_and_qq":     InterfacesAndQualifiedQuantitiesCommand,
            "relationships":         RelationshipsCommand,
            "processor_scalings":    ProcessorScalingsCommand,
            "scale_conversion_v2":   ScaleConversionV2Command,
            "shared_elements":       ExportableElementsCommand,
            "reused_elements":       ImportElementsCommand,
            "import_commands":       NestedCommandsCommand,
            "ref_pedigree_matrices": PedigreeMatricesReferencesCommand,
            "ref_bibliographic":     BibliographicReferencesCommand,
            # "ref_source":            SourceReferencesCommand,
            "ref_geographical":      GeographicReferencesCommand,
            "ref_provenance":        ProvenanceReferencesCommand,
            "indicators":            IndicatorsCommand,
            "problem_statement":     ProblemSolvingCommand
            }
    if cmd_type in cmds:
        tmp = cmds[cmd_type](name)  # Reflective CALL to construct the empty command instance
        tmp._serialization_type = cmd_type  # Injected attribute. Used later for serialization
        tmp._serialization_label = name  # Injected attribute. Used later for serialization
        tmp._source_block_name = source_block
        if isinstance(json_input, (str, dict, list)):
            if json_input != {}:
                issues = tmp.json_deserialize(json_input)
            else:
                issues = []
        else:
            # NO SPECIFICATION
            raise Exception("The command '" + cmd_type + " " + name if name else "<unnamed>" + " does not have a specification.")
        return tmp, issues  # Return the command and the issues found during the deserialization
    else:
        # UNKNOWN COMMAND
        raise Exception("Unknown command type: " + cmd_type)
