"""
Given a State, elaborate an equivalent RDF file which can be exploited outside

"""
from nexinfosys.model_services import State, get_case_study_registry_objects
from owlready2 import *

from nexinfosys.model_services.workspace import execute_file_return_issues
from nexinfosys.models.musiasem_concepts import FactorType, Processor, Factor
from nexinfosys.restful_service.serialization import deserialize_state, serialize_state


def generate_rdf_from_object_model(state: State):
    """
    Using the base ontology "musiasem.owl", generate a file

    :param state:
    :return:
    """
    onto = get_ontology("file:////home/rnebot/Dropbox/nis-backend/nexinfosys/ie_exports/musiasem.owl").load()
    glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
    # onto = get_ontology("file:///musiasem.owl").load()

    # Individuals / instances
    # InterfaceTypes
    fts = glb_idx.get(FactorType.partial_key())
    # Processors
    procs = glb_idx.get(Processor.partial_key())
    # Interfaces and Quantities
    fs = glb_idx.get(Factor.partial_key())

    # Relationships
    #  Scale

    #  ScaleChange
    #  Dependency
    #  Exchange
    #  Part-Of

    # Processor Properties
    #  System (maybe a Class)
    #  Geolocation (maybe a Class)
    #  is_functional (boolean)
    #  is_structural (boolean)
    #  is_notional (boolean)
    #  ...
    # InterfaceType Properties
    #  RoegenType (NOT a class)
    #
    # Interface Properties
    #  Orientation
    #  Quantification (a Class)
    #
    # Relationship Properties
    #  Source (another individual)
    #  Target (another individual)
    #  Weight (property)

    # Formal 2 Semantic


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    input_file = ""
    state_file = ""
    if not os.path.exists(state_file):
        isess, issues = execute_file_return_issues(input_file, generator_type="spreadsheet")
        ensure_dir(state_file)
        # Save state
        s = serialize_state(isess.state)
        with open(state_file, "wt") as f:
            f.write(s)

    with open(state_file, "rt") as f:
        s = f.read()
        state = deserialize_state(s)

    generate_rdf_from_object_model(state)

