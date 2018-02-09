import json

from backend.model_services import IExecutableCommand, State, get_case_study_registry_objects
from backend.model.memory.musiasem_concepts import FactorType, Observer, FactorInProcessorType, \
    Processor, \
    Factor, FactorQuantitativeObservation, QualifiedQuantityExpression, \
    FlowFundRoegenType, ProcessorsSet, HierarchiesSet, allowed_ff_types, ProcessorsRelationPartOfObservation


def obtain_observable_objects(name: str, source: str, state: State,
                              proc_external: bool, proc_attributes: dict, proc_location,
                              fact_roegen_type: FlowFundRoegenType, fact_attributes: dict,
                              fact_incoming: bool, fact_external: bool, fact_location):
    """
    Parse the name, which can specify a processor AND a factor
    Check if the processor exists, if not create it
    Check if the FactorType exists, if not create it
    Check if the Factor exists, if not create it

    :param name:
    :param state:
    :return:
    """
    # TODO Parse the name. List of processor names + list of factor names
    name.split(".")  # TODO
    p_names = []
    f_names = []

    # Get objects from state
    glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)

    # CREATE the Observer for the relations (PART-OF for now)
    oer_key = {"_type": "Observer", "_name": source}
    oer = glb_idx.get(oer_key)
    if not oer:
        oer = Observer(source)
        glb_idx.put(oer_key, oer)
    else:
        oer = oer[0]

    result = None

    # Processors
    parent = None
    acum_name = ""
    for i, p_name in enumerate(p_names):
        last = i == len(p_names-1)

        # CREATE processor(s) (if it does not exist). The processor is an Observable
        acum_name += "." if acum_name != "" else "" + p_name
        p_key = {"_type": "Processor", "_full_name": acum_name, "_name": p_name}
        p = glb_idx.get(p_key)
        if not p:
            attrs = proc_attributes if last else None
            location = proc_location if last else None
            p = Processor(p_name,
                          external=proc_external,
                          location=location,
                          tags=None,
                          attributes=attrs
                          )
            # Index it, adding the attributes only if it is the processor in play
            if last and proc_attributes:
                p_key.update(proc_attributes)
            glb_idx.put(p_key, p)
        else:
            p = p[0]  # First element (list of results, should be length 1)

        result = p

        if parent:
            # Create PART-OF relation
            o1 = ProcessorsRelationPartOfObservation.create_and_append(parent, p, oer)  # Part-of
            # TODO Use full name instead of simple when defining the relation ?????
            # TODO Can UUID be used ?????
            rel_key = {"_type": "ProcessorsRelationPartOfObservation", "_parent": parent.name,
                       "_child": acum_name}
            glb_idx.put(rel_key, o1)

        parent = p

    # FactorTypes and Factor (for the last FactorType)
    parent = None
    acum_name = ""
    for ft_name in f_names:
        last = i == len(f_names-1)

        # CREATE factor type(s) (if it does not exist).
        # The Factor Type is a Class of Observables (it is NOT observable: neither quantities nor relations)
        acum_name += "." if acum_name != "" else "" + p_name
        ft_key = {"_type": "FactorTypen", "_full_name": acum_name, "_name": ft_name}
        ft = glb_idx.get(ft_key)
        if not ft:
            attrs = fact_attributes if last else None
            ft = FactorType(ft_name,  #
                            parent=parent, hierarchy=None,
                            tipe=fact_roegen_type,  #
                            tags=None,  # No tags
                            attributes=attrs,
                            expression=None  # No expression
                            )
            if last and fact_attributes:
                ft_key.update(fact_attributes)
            glb_idx.put(ft_key, ft)
        else:
            ft = ft[0]  # First element (list of results, should be length 1)

        if last:
            # CREATE Factor (if it does not exist). An Observable
            f_key = {"_type": "Factor", "_full_name": acum_name, "_name": ft_name}
            f = glb_idx.get(f_key)
            if not f:
                f = Factor(ft_name,
                           p,
                           in_processor_type=FactorInProcessorType(external=fact_external, incoming=fact_incoming),
                           taxon=ft,
                           location=fact_location,
                           tags=None,
                           attributes=fact_attributes)
                glb_idx.put(f_key, f)
            result = f

        parent = ft

    return result  # Return either a Processor or a Factor, to which Observations can be attached


class StructureCommand(IExecutableCommand):
    """
    It serves to specify the structure connecting processors or flows/funds

    """
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        """
            Process each of the specified relations, creating the endpoints if they do not exist already
            {"name": <processor or factor>,
             "attributes": {"<attr>": "value"},
             "type": <default relation type>,
             "dests": [
                {"name": <processor or factor>,
                 ["type": <relation type>,]
                 "weight": <expression resulting in a numeric value>
                }
             }
        """
        # Process each record
        for o in self._content["origins"]:
            # origin processor[+factor] -> relation (weight) -> destination processor[+factor]
            origin_name = o["name"]
            if "tags" in o:
                origin_tags = o["tags"]
            else:
                origin_tags = None
            source = o["source"]
            # Obtain origin. If it does not exist create it
            origin = obtain_observable_objects(origin_name, source, state,
                              proc_external=None, proc_attributes=origin_tags, proc_location=None,
                              fact_roegen_type=None, fact_attributes=None,
                              fact_incoming=None, fact_external=None, fact_location=None)
            if origin_tags:
                origin.attributes = origin_tags
            for r in o["dests"]:
                destination_name = r["name"]
                rel_type = r["type"]
                weight = r["weight"]  # For flow relations
                # Obtain destination. If it does not exist create it
                destination = obtain_observable_objects(destination_name, source, state,
                                proc_external=None, proc_attributes=origin_tags, proc_location=None,
                                fact_roegen_type=None, fact_attributes=None,
                                fact_incoming=None, fact_external=None, fact_location=None)
                # Add relation(s) (check if duplicated!!)
                rel = obtain_relation(origin, destination, rel_type, weight, state)

        return None, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the content
        return self._content

    def json_deserialize(self, json_input):
        # TODO Check validity
        issues = []
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)
        return issues
