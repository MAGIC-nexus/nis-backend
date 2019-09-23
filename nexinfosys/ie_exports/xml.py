"""
Export processors to XML
Inputs are both the registry and the output dataframe
The registry serves to prepare the structure of the file

"""
from typing import Dict

from nexinfosys import case_sensitive
from nexinfosys.common.helper import strcmp, PartialRetrievalDictionary
from nexinfosys.model_services import State, get_case_study_registry_objects
from nexinfosys.models.musiasem_concepts import ProcessorsRelationPartOfObservation, Processor, Factor


def xml_interface(iface: Factor):
    """

    :param iface:
    :return:
    """
    s = f'<{iface.name} type="{iface.taxon.name}" sphere="{iface.sphere}" ' \
        f'roegen_type="{iface.roegen_type}" orientation="{iface.orientation}" ' \
        f'opposite_processor_type="{iface.opposite_processor_type}" />'
    if case_sensitive:
        return s
    else:
        return s.lower()


def xml_processor(p: Processor, registry: PartialRetrievalDictionary, p_map: Dict[str, Processor]):
    """
    Return the XML of a processor
    Recursive into children

    :param p:
    :return:
    """
    children = p.children(registry)
    full_name = p.full_hierarchy_names(registry)[0]
    if case_sensitive:
        p_map[full_name] = p
    else:
        p_map[full_name.lower()] = p

    s = f"""
<{p.name} fullname="{full_name}" system="{p.processor_system}" subsystem="{p.subsystem_type}" functional="{"true" if strcmp(p.functional_or_structural, "Functional") else "false"}" >
    <interfaces>
    {chr(10).join([xml_interface(f) for f in p.factors])}
    </interfaces>
    {chr(10).join([xml_processor(c, registry, p_map) for c in children])}    
</{p.name}>     
    """
    if case_sensitive:
        return s
    else:
        return s.lower()


def export_model_to_xml(registry: PartialRetrievalDictionary) -> str:
    """
    Elaborate an XML string containing the nested processors and their attributes.
    Also the interfaces inside processors
    :param registry:
    :return:
    """

    # Part of relationships
    por = registry.get(ProcessorsRelationPartOfObservation.partial_key())
    # Keep those affecting Instance processors
    por = [po for po in por if strcmp(po.parent_processor.instance_or_archetype, "Instance")]
    # Get root processors (set of processors not appearing as child_processor)
    parents = set([po.parent_processor for po in por])
    children = set([po.child_processor for po in por])
    roots = parents.difference(children)
    # leaves = children.difference(parents)
    result = ''
    p_map = {}
    for p in roots:
        result += xml_processor(p, registry, p_map)

    return result, p_map

