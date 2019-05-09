"""
Export processors to XML
Inputs are both the registry and the output dataframe
The registry serves to prepare the structure of the file

"""
from backend.common.helper import strcmp, PartialRetrievalDictionary
from backend.model_services import State, get_case_study_registry_objects
from backend.models.musiasem_concepts import ProcessorsRelationPartOfObservation, Processor, Factor


def xml_interface(iface: Factor):
    """

    :param iface:
    :return:
    """
    return f"""
<interface name="{iface.name}" type="{iface.taxon.name}" sphere="{iface.sphere}" roegen_type="{iface.roegen_type}" orientation="{iface.orientation} opposite_processor_type="{iface.opposite_processor_type}">
</interface>     
    """


def xml_processor(p: Processor, registry: PartialRetrievalDictionary):
    """
    Return the XML of a processor
    Recursive into children

    :param p:
    :return:
    """
    children = p.children(registry)
    return f"""
<{p.name} fullname="{p.full_hierarchy_names(registry)[0]}" system="{p.processor_system}" subsystem="{p.subsystem_type}" functional={"true" if strcmp(p.functional_or_structural, "Functional") else "false"} >
    <interfaces>
    {chr(10).join([xml_processor(f) for f in p.factors])}
    </interfaces>
    {chr(10).join([xml_processor(c) for c in children])}    
</{p.name}>     
    """


def export_model_to_xml(state: State) -> str:
    """
    Elaborate an XML string containing the nested processors and their attributes.
    Also the interfaces inside processors
    :param state:
    :return:
    """

    # Registry
    registry, _, _, _, _ = get_case_study_registry_objects(state)
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
    for p in roots:
        result += xml_processor(p)

    return result

