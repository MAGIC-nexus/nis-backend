from backend.model.memory.musiasem_concepts import Heterarchy, Taxon, Processor, FactorTaxon


def build_hierarchy(name, type_name, registry, h: dict, hierarchy=None, level_names=None):
    """
    :param name: Name of the hierarchy or None if recursive call
    :param type_name: "Taxon", "Processor", "FactorTaxon"
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
        elif type_name.lower() == "factortaxon":
            t = FactorTaxon(k, None, hie)

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