from typing import Union, List, Tuple, Optional

from backend.common.helper import PartialRetrievalDictionary, strcmp
from backend.model_services import State, get_case_study_registry_objects
from backend.model.memory.musiasem_concepts import \
    FlowFundRoegenType, FactorInProcessorType, RelationClassType, allowed_ff_types, \
    Processor, FactorType, Observer, Factor, \
    ProcessorsRelationPartOfObservation, ProcessorsRelationUndirectedFlowObservation, \
    ProcessorsRelationUpscaleObservation, \
    FactorsRelationDirectedFlowObservation, Heterarchy, Taxon, QualifiedQuantityExpression, \
    FactorQuantitativeObservation


def find_observable(name: str, idx: PartialRetrievalDictionary, processor: Processor = None,
                    factor_type: FactorType = None) -> Union[Factor, Processor, FactorType]:
    """
    From the processor:factortype, obtain the Factor, searching in the INDEX of objects,
    first find the Processor, second the FactorType, finally the Factor
    :param name: ":" separated processor name and factor type name. "p:ft" returns a Factor. "p" or "p:" returns a Processor. ":ft" returns a FactorType
    :param idx: The PartialRetrievalDictionary where the objects are indexed
    :param processor: Already resolved Processor. If ":ft" is specified, it will use this parameter to return a Factor (not a FactorType)
    :param factor_type: Already resolved FactorType. If "p:" is specified, it will use this parameter to return a Factor (not a Processor)
    :return: Processor or FactorType or Factor
    """
    res = None
    if isinstance(name, str):
        s = name.split(":")
        if len(s) == 2:  # There is a ":"
            p_name = s[0]
            f_name = s[1]
            if not p_name:  # Processor can be blank
                p_name = None
            if not f_name:  # Factor type can be blank
                f_name = None
        elif len(s) == 1:  # If no ":", go just for the processor
            p_name = s[0]
            f_name = None
        # Retrieve the processor
        if p_name:
            p = idx.get(Processor.partial_key(name=p_name, registry=idx))
            if p:
                p = p[0]
        elif processor:
            p = processor
        else:
            p = None

        # Retrieve the FactorType
        if f_name:
            ft = idx.get(FactorType.partial_key(name=f_name, registry=idx))
            if ft:
                ft = ft[0]
        elif factor_type:
            ft = factor_type
        else:
            res = p
            ft = None

        if not p_name and not p:  # If no Processor available at this point, FactorType is being requested, return it
            res = ft
        elif not res and p and ft:
            f = idx.get(Factor.partial_key(processor=p, taxon=ft, registry=idx))
            if f:
                res = f[0]
    else:
        res = name

    return res


def find_or_create_observable(state: Union[State, PartialRetrievalDictionary],
                              name: str, source: Union[str, Observer]=Observer.no_observer_specified,
                              aliases: str=None,  # "name" (processor part) is an alias of "aliases" Processor
                              proc_external: bool=None, proc_attributes: dict=None, proc_location=None,
                              fact_roegen_type: FlowFundRoegenType=None, fact_attributes: dict=None,
                              fact_incoming: bool=None, fact_external: bool=None, fact_location=None):
    """
    Find or create Observables: Processors, Factor and FactorType objects
    It can also create an Alias for a Processor if the name of the aliased Processor is passed (parameter "aliases")

    "name" is parsed, which can specify a processor AND a factor, both hierarchical ('.'), separated by ":"

    :param state:
    :param name: Full name of processor, processor':'factor or ':'factor
    :param source: Name of the observer or Observer itself (used only when creating nested Processors, because it implies part-of relations)
    :param aliases: Full name of an existing processor to be aliased by the processor part in "name"
    :param proc_external: True=external processor; False=internal processor
    :param proc_attributes: Dictionary with attributes to be added to the processor if it is created
    :param proc_location: Specification of where the processor is physically located, if it applies
    :param fact_roegen_type: Flow or Fund
    :param fact_attributes: Dictionary with attributes to be added to the Factor if it is created
    :param fact_incoming: True if the Factor is incoming regarding the processor; False if it is outgoing
    :param fact_external: True if the Factor comes from Environment
    :param fact_location: Specification of where the processor is physically located, if it applies
    :return: Processor, FactorType, Factor
    """

    # Decompose the name
    p_names, f_names = _obtain_name_parts(name)

    # Get objects from state
    if isinstance(state, PartialRetrievalDictionary):
        glb_idx = state
    else:
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)

    # Get the Observer for the relations (PART-OF for now)
    if source:
        if isinstance(source, Observer):
            oer = source
        else:
            oer = glb_idx.get(Observer.partial_key(name=source, registry=glb_idx))
            if not oer:
                oer = Observer(source)
                glb_idx.put(oer.key(registry=glb_idx), oer)
            else:
                oer = oer[0]

    result = None
    p = None  # Processor to which the Factor is connected
    ft = None  # FactorType
    f = None  # Factor

    if p_names and aliases:
        # Create an alias for the Processor
        if isinstance(aliases, str):
            p = glb_idx.get(Processor.partial_key(aliases, registry=glb_idx))
        elif isinstance(aliases, Processor):
            p = aliases
        if p:
            full_name = ".".join(p_names)
            # Look for a processor named <full_name>, it will be an AMBIGUITY TO BE AVOIDED
            p1, k1 = glb_idx.get(Processor.partial_key(full_name), True)
            if p1:
                # If it is an ALIAS, void the already existing because there would be no way to decide
                # which of the two (or more) do we need
                if Processor.is_alias_key(k1[0]):
                    # Assign NONE to the existing Alias
                    glb_idx.put(k1[0], None)
            else:
                # Create the ALIAS
                k_ = Processor.alias_key(full_name, p)
                glb_idx.put(k_, p)  # An alternative Key pointing to the same processor
    else:
        # Find or create the "lineage" of Processors, using part-of relation ("matryoshka" or recursive containment)
        parent = None
        acum_name = ""
        for i, p_name in enumerate(p_names):
            last = i == (len(p_names)-1)

            # CREATE processor(s) (if it does not exist). The processor is an Observable
            acum_name += ("." if acum_name != "" else "") + p_name
            p = glb_idx.get(Processor.partial_key(name=acum_name))
            if not p:
                attrs = proc_attributes if last else None
                location = proc_location if last else None
                p = Processor(acum_name,
                              external=proc_external,
                              location=location,
                              tags=None,
                              attributes=attrs
                              )
                # Index it, adding the attributes only if it is the processor in play
                p_key = p.key(glb_idx)
                if last and proc_attributes:
                    p_key.update({k: ("" if v is None else v) for k, v in proc_attributes.items()})
                glb_idx.put(p_key, p)
            else:
                p = p[0]

            result = p

            if parent:
                # Create PART-OF relation
                o1 = glb_idx.get(ProcessorsRelationPartOfObservation.partial_key(parent=parent, child=p, registry=glb_idx))
                if not o1:
                    o1 = ProcessorsRelationPartOfObservation.create_and_append(parent, p, oer)  # Part-of
                    glb_idx.put(o1.key(registry=glb_idx), o1)

            parent = p

    # Find or create the lineage of FactorTypes and for the last FactorType, find or create Factor
    parent = None
    acum_name = ""
    for i, ft_name in enumerate(f_names):
        last = i == len(f_names)-1

        # CREATE factor type(s) (if it does not exist). The Factor Type is a Class of Observables (it is NOT observable: neither quantities nor relations)
        acum_name += ("." if acum_name != "" else "") + ft_name
        ft = glb_idx.get(FactorType.partial_key(name=acum_name, registry=glb_idx))
        if not ft:
            attrs = fact_attributes if last else None
            ft = FactorType(acum_name,  #
                            parent=parent, hierarchy=None,
                            tipe=fact_roegen_type,  #
                            tags=None,  # No tags
                            attributes=attrs,
                            expression=None  # No expression
                            )
            ft_key = ft.key(glb_idx)
            if last and fact_attributes:
                ft_key.update(fact_attributes)
            glb_idx.put(ft_key, ft)
        else:
            ft = ft[0]

        if last and p:  # The Processor must exist. If not, nothing is created or obtained
            # CREATE Factor (if it does not exist). An Observable
            f = glb_idx.get(Factor.partial_key(processor=p, taxon=ft, registry=glb_idx))
            if not f:
                f = Factor(acum_name,
                           p,
                           in_processor_type=FactorInProcessorType(external=fact_external, incoming=fact_incoming),
                           taxon=ft,
                           location=fact_location,
                           tags=None,
                           attributes=fact_attributes)
                glb_idx.put(f.key(glb_idx), f)
            else:
                f = f[0]

            result = f

        parent = ft

    return p, ft, f  # Return all the observables (some may be None)
    # return result  # Return either a Processor or a Factor, to which Observations can be attached


def create_quantitative_observation(state: Union[State, PartialRetrievalDictionary],
                                    factor: Union[str, Factor],
                                    value: str, unit: str,
                                    observer: Union[str, Observer]=Observer.no_observer_specified,
                                    spread: str=None, assessment: str=None, pedigree: str=None, pedigree_template: str=None,
                                    relative_to: Union[str, Factor]=None,
                                    time: str=None,
                                    geolocation: str=None,
                                    comments: str=None,
                                    tags=None, other_attributes=None,
                                    proc_aliases: str=None,
                                    proc_external: bool=None, proc_attributes: dict=None, proc_location=None,
                                    ftype_roegen_type: FlowFundRoegenType=None, ftype_attributes: dict=None,
                                    fact_incoming: bool=None, fact_external: bool=None, fact_location=None):
    """
    Creates an Observation of a Factor
    If the Factor does not exist, it is created
    If no "value" is passed, only the Factor is created

    :param state:
    :param factor: string processor:factor_type or Factor
    :param value: expression with the value
    :param unit: metric unit
    :param observer: string with the name of the observer or Observer
    :param spread: expression defining uncertainty of :param value
    :param assessment:
    :param pedigree: encoded assessment of the quality of the science/technique of the observation
    :param pedigree_template: reference pedigree matrix used to encode the pedigree
    :param relative_to: Factor Type in the same Processor to which the value is relative
    :param time: time extent in which the value is valid
    :param geolocation: where the observation is
    :param comments: open comments about the observation
    :param tags: list of tags added to the observation
    :param other_attributes: dictionary added to the observation
    :param proc_aliases: name of aliased processor (optional). Used only if the Processor does not exist
    :param proc_external: True if the processor is outside the case study borders, False if it is inside. Used only if the Processor does not exist
    :param proc_attributes: Dictionary with attributes added to the Processor. Used only if the Processor does not exist
    :param proc_location: Reference specifying the location of the Processor. Used only if the Processor does not exist
    :param ftype_roegen_type: Either FUND or FLOW (applied to FactorType). Used only if the FactorType does not exist
    :param ftype_attributes: Dictionary with attributes added to the FactorType. Used only if the FactorType does not exist
    :param fact_incoming: Specifies if the Factor goes into or out the Processor. Used if the Factor (not FactorType) does not exist
    :param fact_external: Specifies if the Factor is injected from an external Processor. Used if the Factor (not FactorType) does not exist
    :param fact_location: Reference specifying the location of the Factor. Used if the Factor does not exist

    :return:
    """
    # Get objects from state
    if isinstance(state, State):
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
    elif isinstance(state, PartialRetrievalDictionary):
        glb_idx = state

    # Obtain factor
    p, ft = None, None
    if not isinstance(factor, Factor):
        p, ft, factor_ = find_or_create_observable(state,
                                                   factor,
                                                   # source=None,
                                                   aliases=proc_aliases,
                                                   proc_external=proc_external,
                                                   proc_attributes=proc_attributes,
                                                   proc_location=proc_location,
                                                   fact_roegen_type=ftype_roegen_type,
                                                   fact_attributes=ftype_attributes,
                                                   fact_incoming=fact_incoming,
                                                   fact_external=fact_external,
                                                   fact_location=fact_location
                                                   )
        if not isinstance(factor_, Factor):
            raise Exception("The name specified for the factor ('"+factor+"') did not result in the obtention of a Factor")
        else:
            factor = factor_

    if value:
        # Get the Observer for the relations (PART-OF for now)
        if isinstance(observer, Observer):
            oer = observer
        else:
            oer = glb_idx.get(Observer.partial_key(name=observer, registry=glb_idx))
            if not oer:
                oer = Observer(observer)
                glb_idx.put(oer.key(registry=glb_idx), oer)
            else:
                oer = oer[0]

        # Create the observation
        o = _create_quantitative_observation(factor,
                                             value, unit, spread, assessment, pedigree, pedigree_template,
                                             oer,
                                             relative_to,
                                             time,
                                             geolocation,
                                             comments,
                                             tags, other_attributes
                                             )
        # Register
        glb_idx.put(o.key(glb_idx), o)

        # Return the observation
        return p, ft, factor, o
    else:
        # Return the Factor
        return p, ft, factor, None


def create_relation_observations(state: State,
                                 origin: Union[str, Processor, Factor],
                                 destinations: List[Tuple[Union[str, Processor, Factor], Optional[Union[RelationClassType, str]]]],
                                 relation_class: RelationClassType=None,
                                 oer: Union[str, Observer]=Observer.no_observer_specified) -> List:
    """
    Create and register one or more relations from a single origin to one or more destinations.
    Relation parameters (type and weight) can be specified for each destination, or a default relation class parameter is used
    Relation are assigned to the observer "oer"

    :param state: Registry of all objects
    :param origin: Origin of the relation as string, Processor or Factor
    :param destinations: List of tuples, where each tuple can be of a single element, the string, Processor or Factor, or can be accompanied by the relation parameters tuple, with up to two elements, first the relation type, second the string describing the weight
    :param relation_class: Default relation class
    :param oer: str or Observer for the Observer to which relation observations are accounted
    :return: The list of relations
    """
    def get_requested_object(p_, ft_, f_):
        if p_ and not ft_ and not f_:
            return p_
        elif not p_ and ft_ and not f_:
            return ft_
        elif p_ and ft_ and f_:
            return f_

    glb_idx, _, _, _, _ = get_case_study_registry_objects(state)
    # Origin
    p, ft, f = find_or_create_observable(glb_idx, origin)
    origin_obj = get_requested_object(p, ft, f)

    rels = []

    if oer:
        if isinstance(oer, str):
            oer = glb_idx.get(Observer.partial_key(name=oer, registry=glb_idx))
            if not oer:
                oer = Observer(oer)
                glb_idx.put(oer.key(registry=glb_idx), oer)
            else:
                oer = oer[0]

    if not isinstance(destinations, list):
        destinations = [destinations]
    for dst in destinations:
        if not isinstance(dst, tuple):
            dst = tuple([dst])
        # Destination
        dst_obj = None
        if isinstance(origin_obj, Processor) and relation_class == RelationClassType.pp_part_of:
            # Find dst[0]. If it does not exist, create dest UNDER (hierarchically) origin
            dst_obj = find_observable(dst[0], glb_idx)
            if not dst_obj:
                name = origin_obj.full_hierarchy_names(glb_idx)[0] + "." + dst[0]
                p, ft, f = find_or_create_observable(glb_idx, name, source=oer)
                dst_obj = get_requested_object(p, ft, f)
                rel = glb_idx.get(ProcessorsRelationPartOfObservation.partial_key(parent=origin_obj, child=dst_obj, observer=oer))
                if rel:
                    rels.append(rel[0])
                continue  # Skip the rest of the loop
            else:
                dst_obj = dst_obj[0]

        if not dst_obj:
            p, ft, f = find_or_create_observable(glb_idx, dst[0])
            dst_obj = get_requested_object(p, ft, f)
            # If origin is Processor and destination is Factor, create Factor in origin (if it does not exist). Or viceversa
            if isinstance(origin_obj, Processor) and isinstance(dst_obj, Factor):
                # Obtain full origin processor name
                names = origin_obj.full_hierarchy_names(glb_idx)
                p, ft, f = find_or_create_observable(glb_idx, names[0] + ":" + dst_obj.taxon.name)
                origin_obj = get_requested_object(p, ft, f)
            elif isinstance(origin_obj, Factor) and isinstance(dst_obj, Processor):
                names = dst_obj.full_hierarchy_names(glb_idx)
                p, ft, f = find_or_create_observable(glb_idx, names[0] + ":" + origin_obj.taxon.name)
                dst_obj = get_requested_object(p, ft, f)
            # Relation class
            if len(dst) > 1:
                rel_type = dst[1]
            else:
                if not relation_class:
                    if isinstance(origin_obj, Processor) and isinstance(dst_obj, Processor):
                        relation_class = RelationClassType.pp_undirected_flow
                    else:
                        relation_class = RelationClassType.ff_directed_flow
                rel_type = relation_class
            if len(dst) > 2:
                weight = dst[2]
            else:
                weight = ""  # No weight, it only can be used to aggregate
            rel = _obtain_relation(origin_obj, dst_obj, rel_type, oer, weight, glb_idx)
        rels.append(rel)

    return rels

# ########################################################################################
# Auxiliary functions
# ########################################################################################


def _obtain_name_parts(n):
    """
    Parse the name. List of processor names + list of factor names
    :param n:
    :return:
    """
    r = n.split(":")
    if len(r) > 1:
        full_p_name = r[0]
        full_f_name = r[1]
    else:
        full_p_name = r[0]
        full_f_name = ""
    p_ = full_p_name.split(".")
    f_ = full_f_name.split(".")
    if len(p_) == 1 and not p_[0]:
        p_ = []
    if len(f_) == 1 and not f_[0]:
        f_ = []
    return p_, f_


def _create_quantitative_observation(factor: Factor,
                                     value: str, unit: str,
                                     spread: str, assessment: str, pedigree: str, pedigree_template: str,
                                     observer: Observer,
                                     relative_to: Union[str, Factor],
                                     time: str,
                                     geolocation: str,
                                     comments: str,
                                     tags, other_attributes):
    if other_attributes:
        attrs = other_attributes.copy()
    else:
        attrs = {}
    if relative_to:
        if isinstance(relative_to, str):
            rel2 = relative_to
        else:
            rel2 = relative_to.name
    else:
        rel2 = None
    attrs.update({"relative_to": rel2,
                  "time": time,
                  "geolocation": geolocation,
                  "spread": spread,
                  "assessment": assessment,
                  "pedigree": pedigree,
                  "pedigree_template": pedigree_template,
                  "comments": comments
                  }
                 )

    fo = FactorQuantitativeObservation.create_and_append(v=QualifiedQuantityExpression(value + " " + unit),
                                                         factor=factor,
                                                         observer=observer,
                                                         tags=tags,
                                                         attributes=attrs
                                                         )
    return fo


# def find_observable(o, idx: PartialRetrievalDictionary):
#     """
#     Find an observable (Processor, FactorType, Factor) using the appropriate PartialRetrievalDictionary
#
#     :param o: string or directly the object
#     :param idx:
#     :return: Object or None
#     """
#     res = None
#     if isinstance(o, str):
#         # "o" is the name of the object to be searched
#         p_names, f_names = obtain_name_parts(o)
#         if f_names:
#             ft = idx.get(FactorType.partial_key(name=".".join(f_names)))
#         if p_names:
#             p = idx.get(Processor.partial_key(name=".".join(p_names)))
#             # If the search failed, use a step-by-step (relative) naming for the processors part
#             if not p:
#                 for p_name in p_names:
#                     if not p: # First iteration
#                         p = idx.get(Processor.partial_key(name=p_name))
#                     else:
#                         rels = idx.get(ProcessorsRelationPartOfObservation.partial_key(parent=p))
#                         p = None
#                         for r in rels:
#                             simple_name = r.child.name.split(".")[-1]
#                             if strcmp(p_name, simple_name):
#                                 p = r.child
#                                 break
#                         if not p:  # Not found, failed -> exit loop
#                             break
#
#             if f_names:
#                 if p and ft:
#                     res = idx.get(Factor.partial_key(processor=p, taxon=ft, registry=idx))
#                 else:
#                     res = None
#             else:
#                 res = p
#         elif f_names:
#             res = ft
#     else:
#         res = o
#
#     return res


def _get_observer(observer: Union[str, Observer], idx: PartialRetrievalDictionary) -> Observer:
    res = None
    if isinstance(observer, Observer):
        res = observer
    else:
        oer = idx.get(Observer.partial_key(name=observer, registry=idx))
        if oer:
            res = oer[0]
    return res


def _obtain_relation(origin, destination, rel_type: RelationClassType, oer: Union[Observer, str], weight: str, state: Union[State, PartialRetrievalDictionary]):
    """
    Construct and register a relation between origin and destination

    :param origin: Either processor or factor
    :param destination: Either processor or factor
    :param rel_type:
    :param oer: Observer, as object or string
    :param weight: For flow relations
    :param state: State or PartialRetrievalDictionary
    :return: The relation observation
    """
    # Get objects from state
    if isinstance(state, State):
        glb_idx, _, _, _, _ = get_case_study_registry_objects(state)
    elif isinstance(state, PartialRetrievalDictionary):
        glb_idx = state

    # CREATE the Observer for the relation
    if oer and isinstance(oer, str):
        oer = glb_idx.get(Observer.partial_key(name=oer, registry=glb_idx))
        if not oer:
            oer = Observer(oer)
            glb_idx.put(oer.key(registry=glb_idx), oer)
        else:
            oer = oer[0]

    r = None
    if rel_type == RelationClassType.pp_part_of:
        if isinstance(origin, Processor) and isinstance(destination, Processor):
            # Find or Create the relation
            r = glb_idx.get(ProcessorsRelationPartOfObservation.partial_key(parent=origin, child=destination))
            if not r:
                r = ProcessorsRelationPartOfObservation.create_and_append(origin, destination, oer)  # Part-of
                glb_idx.put(r.key(glb_idx), r)
            else:
                r = r[0]
    elif rel_type == RelationClassType.pp_undirected_flow:
        if isinstance(origin, Processor) and isinstance(destination, Processor):
            # Find or Create the relation
            r = glb_idx.get(ProcessorsRelationUndirectedFlowObservation.partial_key(source=origin, target=destination))
            if not r:
                r = ProcessorsRelationUndirectedFlowObservation.create_and_append(origin, destination, oer)  # Undirected flow
                glb_idx.put(r.key(glb_idx), r)
            else:
                r = r[0]
    elif rel_type == RelationClassType.pp_upscale:
        if isinstance(origin, Processor) and isinstance(destination, Processor):
            # Find or Create the relation
            r = glb_idx.get(ProcessorsRelationUpscaleObservation.partial_key(parent=origin, child=destination))
            if not r:
                r = ProcessorsRelationUpscaleObservation.create_and_append(origin, destination, oer, weight)  # Upscale
                glb_idx.put(r.key(glb_idx), r)
            else:
                r = r[0]
    elif rel_type in (RelationClassType.ff_directed_flow, RelationClassType.ff_reverse_directed_flow):
        if isinstance(origin, Factor) and isinstance(destination, Factor):
            if rel_type == RelationClassType.ff_reverse_directed_flow:
                origin, destination = destination, origin
                if weight:
                    weight = "1/("+weight+")"
            # Find or Create the relation
            r = glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key(source=origin, target=destination))
            if not r:
                r = FactorsRelationDirectedFlowObservation.create_and_append(origin, destination, oer, weight)  # Directed flow
                glb_idx.put(r.key(glb_idx), r)
            else:
                r = r[0]

    return r


def build_hierarchy(name, type_name, registry, h: dict, hierarchy=None, level_names=None):
    """
    :param name: Name of the hierarchy or None if recursive call
    :param type_name: "Taxon", "Processor", "FactorType"
    :param registry: Use a registry both to retrieve existing names (member of several taxonomies) and to register them
    :param h: Dictionary, with nested dictionaries
    :param hierarchy: Hierarchy. Do not use!!
    :return: The hierarchy. It can also return a list of hierarchy nodes (but for the recursive internal use only)
    """
    if hierarchy:
        hie = hierarchy
    elif name:
        hie = Heterarchy(name)

    if registry:
        # TODO Register the new object with its name
        pass

    c = []
    for k in h:
        if type_name.lower() == "taxon":
            t = Taxon(k, None, hie)
        elif type_name.lower() == "processor":
            t = Processor(k, None, hie)
        elif type_name.lower() == "factortype":
            t = FactorType(k, None, hie)

        if registry:
            # TODO Register the new object with its name
            pass

        if h[k]:
            c2 = build_hierarchy(None, type_name, registry, h[k], hie)
            for i in c2:
                i.set_parent(t, hie)
        c.append(t)

    if name:
        hie.roots_append(c)
        if level_names:
            hie.level_names = level_names

        return hie
    else:
        return c