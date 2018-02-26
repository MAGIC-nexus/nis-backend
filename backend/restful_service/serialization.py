from typing import Union

from sqlalchemy.orm import relationship, backref, composite, scoped_session, sessionmaker, class_mapper
import pandas as pd

# Some ideas from function "model_to_dict" (Google it, StackOverflow Q&A)
from backend.common.helper import PartialRetrievalDictionary
from backend.model.persistent_db.persistent import serialize_from_object, deserialize_to_object
from backend.model_services import State, get_case_study_registry_objects


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


def serialize_state(state: State):
    """
    Serialization prepared for a given organization of the state

    !!! WARNING: It destroys "state"

    :return:
    """

    def serialize_dataframe(df):
        return df.index.names, df.to_dict()

    # Iterate all namespaces
    for ns in state.list_namespaces():
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state, ns)
        if glb_idx:
            glb_idx = glb_idx.to_pickable()
            state.set("_glb_idx", glb_idx, ns)
        # TODO Serialize other DataFrames.
        # Datasets
        for ds_name in datasets:
            ds = datasets[ds_name]
            if isinstance(ds.data, pd.DataFrame):
                ds.data = serialize_dataframe(ds.data)

    return serialize_from_object(state)


def deserialize_state(st: str):
    """
    Deserializes an object previously serialized using "serialize_state"

    It can receive also a "State" modified for the serialization to restore it

    :param st:
    :return:
    """
    def deserialize_dataframe(t):
        df = pd.DataFrame(t[1])
        df.index.names = t[0]
        return df

    if isinstance(st, str):
        state = deserialize_to_object(st)
    else:
        raise Exception("It must be a string")

    # Iterate all namespaces
    for ns in state.list_namespaces():
        glb_idx = state.get("_glb_idx", ns)
        if isinstance(glb_idx, dict):
            glb_idx = PartialRetrievalDictionary().from_pickable(glb_idx)
            state.set("_glb_idx", glb_idx)
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state, ns)
        # TODO Deserialize DataFrames
        # In datasets
        for ds_name in datasets:
            ds = datasets[ds_name]
            if ds.data:
                ds.data = deserialize_dataframe(ds.data)
    return state
