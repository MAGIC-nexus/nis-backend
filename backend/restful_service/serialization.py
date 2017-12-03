from sqlalchemy.orm import relationship, backref, composite, scoped_session, sessionmaker, class_mapper

# Some ideas from function "model_to_dict" (Google it, StackOverflow Q&A)


def serialize(o_list):
    """
    Receives a list of SQLAlchemy objects to serialize
    The objects can be related between them by OneToMany and ManyToOne relations
    Returns a list of dictionaries with their properties and two special
    properties, "_nis_class_name" and "_nis_object_id" allowing the reconstruction

    Raise exception if some of the objects refers to an object OUT of the graph

    :param o_list:
    :return:
    """

    def fullname(o):
        return o.__module__ + "." + o.__class__.__name__

    # Dictionary to map obj to ID
    d_ref = {o: i for i, o in enumerate(o_list)}

    # Expand the list to referred objects
    cont = len(o_list)
    proc_lst = o_list
    while True:
        o_list2 = []
        for i, o in enumerate(proc_lst):
            # Relationships
            relationships = [(name, rel) for name, rel in class_mapper(o.__class__).relationships.items()]
            for name, relation in relationships:
                if str(relation.direction) != "symbol('ONETOMANY')":
                    ref_obj = o.__dict__.get(name)
                    if ref_obj:
                        if ref_obj not in d_ref:
                            d_ref[ref_obj] = cont
                            o_list2.append(ref_obj)
                            cont += 1

        o_list.extend(o_list2)
        proc_lst = o_list2
        if len(o_list2) == 0:
            break

    # Do the transformation to list of dictionaries
    d_list = []
    for i, o in enumerate(o_list):
        d = {c.key: getattr(o, c.key) for c in o.__table__.columns}
        d["_nis_class_name"] = fullname(o)
        d["_nis_object_id"] = i
        # Relationships
        relationships = [(name, rel) for name, rel in class_mapper(o.__class__).relationships.items()]
        for name, relation in relationships:
            if str(relation.direction) != "symbol('ONETOMANY')":
                ref_obj = o.__dict__.get(name)
                if ref_obj:
                    if ref_obj in d_ref:
                        d[name] = d_ref[ref_obj]
                else:
                    d[name] = -1  # None

        d_list.append(d)

    return d_list


def deserialize(d_list):
    """
    Receives a list of dictionaries representing SQLAlchemy object previously serialized

    :param d_list:
    :return:
    """
    def instantiate(full_class_name: str, c_dict: dict):
        import importlib
        if full_class_name in c_dict:
            class_ = c_dict[full_class_name]
        else:
            module_name, class_name = full_class_name.rsplit(".", 1)
            class_ = getattr(importlib.import_module(module_name), class_name)
            c_dict[full_class_name] = class_

        return class_()

    o_list = []
    c_list = {}  # Dictionary of classes (full_class_name, class)
    # Construct instances
    for d in d_list:
        o_list.append(instantiate(d["_nis_class_name"], c_list))
    # Now populate them
    ids = []
    for i, d in enumerate(d_list):
        o = o_list[i]
        o.__dict__.update({c.key: d[c.key] for c in o.__table__.columns})
        # Relationships
        relationships = [(name, rel) for name, rel in class_mapper(o.__class__).relationships.items()]
        for name, relation in relationships:
            if str(relation.direction) != "symbol('ONETOMANY')":
                ref_idx = d[name]
                if ref_idx < 0:
                    setattr(o, name, None)
                else:
                    setattr(o, name, o_list[ref_idx])
                    for t in relation.local_remote_pairs:
                        o_id_name = t[0].name  # Or t[0].key
                        r_id_name = t[1].name  # Or t[1].key
                        ids.append((o, o_id_name, o_list[ref_idx], r_id_name))

    for t in ids:
        k = getattr(t[2], t[3])
        if k:
            setattr(t[0], t[1], k)

    return o_list
