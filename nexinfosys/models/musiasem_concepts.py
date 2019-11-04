# -*- coding: utf-8 -*-

"""
Model of MuSIASEM concepts

The main meta-concepts are

* Observable. Something on which hypothesis can be performed
* Observation. A specific hypothesis, in our case in the form of a simple fact, like a qualified quantity or a relation between two or more Observables
* Observer. An agent producing observations on observables

The main MuSIASEM concepts are

* Processor. A Processor in MuSIASEM 2.0
* FactorType. A type of flow or fund
* Factor. An instance of FactorType associated to a specific processor

The three are Observables, so Observations can be attached to them. FactorType allow a single Observer for their Relations.

Support concepts

* Hierarchy. A set of categories organized either as a list or as a tree, or as a list of trees (forest?)
* Mapping. Defines how to obtain a set of "destination categories" from another set of "origin categories"
  * Currently it is Many to One
  * Many to Many is being considered
  * Because mappings are unidirectional (origin to destination), transitivity is implicit
  * Mapping of several sets of categories is also being considered (will be useful?)
* Observations:
  * QualifiedQuantity. Attached to Factors
  * Relation. Different kinds: part-of (holoarchy), flow (directed or undirected), upscale, transform (for FactorTypes)
* Parameter.
* Indicator. An arithmetic expression embedding some semantics, combining observations on Factors
* NUSAP. Materialized in a QualifiedQuantity
  * Number
  * Unit
  * Spread
  * Assessment?
  * PedigreeTemplate. Allows specifying the Pedigree of a QualifiedQuantity
* Geolocation
* TimeExtent
* ObservationProcess, Source. Is the Observer
*

=============
= About RDF =
=============
The most flexible approach involves using RDF, RDFlib.
An interesting paper: "A survey of RDB to RDF translation approaches and tools"

"""

import collections
import json
import logging
from collections import OrderedDict
from enum import Enum
from typing import *  # Type hints

import pandas as pd
import pint  # Physical Units management

from nexinfosys.common.helper import create_dictionary, strcmp, PartialRetrievalDictionary, \
    case_sensitive, is_boolean, is_integer, is_float, is_datetime, is_url, is_uuid, to_datetime, to_integer, to_float, \
    to_url, to_uuid, to_boolean, to_category, to_str, is_category, is_str, is_geo, to_geo, ascii2uuid, \
    Encodable, name_and_id_dict, ifnull, FloatOrString, UnitConversion
from nexinfosys.model_services import State, get_case_study_registry_objects, LocallyUniqueIDManager
from nexinfosys.models import CodeImmutable
from nexinfosys.models import ureg, log_level

logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# #################################################################################################################### #
# PRELIMINARY TYPES
# #################################################################################################################### #


class FlowFundRoegenType(Enum):  # Used in FlowFund
    flow = 1
    fund = 0


FactorInProcessorType = collections.namedtuple("FactorInProcessorType", "external incoming")

allowed_ff_types = ["int_in_flow",  "int_in_fund",  "ext_in_fund", "int_out_flow", "ext_in_flow",
                    "ext_out_flow", "env_out_flow", "env_in_flow", "env_in_fund"
                    ]

# #################################################################################################################### #
# BASE CLASSES
# #################################################################################################################### #


class Identifiable(Encodable):
    """
    A concept with a unique ID (UUID). Which makes it an unambiguously addressed Entity
    The UUID is obtained in base 85 (a more compact ASCII representation)
    """
    def __init__(self):
        self._id = LocallyUniqueIDManager().get_new_id()

    def encode(self):
        return {
            "id": self.uuid
        }

    @property
    def ident(self):
        return self._id

    @property
    def uuid(self):
        return ascii2uuid(self._id)


class Nameable(Encodable):
    """ Concepts with name. Almost all. """
    def __init__(self, name):
        self._name = name

    def encode(self):
        return {
            "name": self.name
        }

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n: str):
        self._name = n


class Taggable:
    """ A concept with a set of tags """
    def __init__(self, tags):
        self._tags = set()
        if tags:
            self.tags_append(tags)

    @property
    def tags(self):
        return self._tags

    def tags_append(self, taxon):
        # TODO Taxon should be a leaf node. It appears the need to specify to which hierarchies it is leaf node
        if isinstance(taxon, (list, set)):
            lst = taxon
        else:
            lst = [taxon]

        self._tags.update(lst)


class Automatable(Encodable):
    """ A concept that could have been generated by an automatic process.
        A flag, the producer object and a reason. """
    def __init__(self):
        self._automatically_generated = False
        self._producer = None  # Object (instance) responsible of the production
        self._generation_reason = None  # Clone, automatic reasoning, solving, ...

    def encode(self):
        return {
            'automatically_generated': self.automatically_generated,
            'producer': self.producer,
            'generation_reason': self.automatic_generation_reason
        }

    @property
    def automatically_generated(self):
        return self._automatically_generated

    @automatically_generated.setter
    def automatically_generated(self, aut: bool):
        self._automatically_generated = aut

    @property
    def producer(self):
        return self._producer

    @producer.setter
    def producer(self, p):
        self._producer = p

    @property
    def automatic_generation_reason(self):
        return self._generation_reason

    @automatic_generation_reason.setter
    def automatic_generation_reason(self, reason):
        self._generation_reason = reason


attribute_types = [("Boolean", is_boolean, to_boolean),
                   ("Integer", is_integer, to_integer),
                   ("Float", is_float, to_float),
                   ("Datetime", is_datetime, to_datetime),
                   ("URL", is_url, to_url),
                   ("Geo", is_geo, to_geo),
                   ("UUID", is_uuid, to_uuid),
                   ("Category", is_category, to_category),
                   ("String", is_str, to_str)
                   ]


class AttributeType(Nameable, Identifiable):
    """
    Defines an attribute type, which may be used in different entity types and instances of these types
    """
    def __init__(self, name, atype: str, description: str=None, domain: str=None, element_types: List[str]=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        self._atype = atype  # One of "attribute_types": Number, Boolean, URL, UUID, Datetime, String, Category
        self._element_types = element_types  # A list of elements to which it can be applied: Parameter, Processor, InterfaceType, Interface, ... If empty, can be applied to any type
        self._domain = domain  # Formal definition of possible values for the attribute
        self._description = description

    @staticmethod
    def partial_key(name: str=None, ident: str=None):
        d = dict(_t="at")
        if name:
            d["_n"] = name
        if ident:
            d["__id"] = ident
        return d

    def key(self):
        return {"_t": "at", "_n": self.name, "__id": self.ident}


def convert_and_infer_attribute_type(v: str):
    """
    Given a string obtain the type and the value in that type

    :param v:
    :return: tuple composed by the value and the value type
    """
    v = v.strip()
    for t in attribute_types:
        if t[1](v):
            return t[2](v), t[0]


class Qualifiable(Encodable):
    """
    Add properties to another class, which can be accessed through the new attribute "attributes".
    Some properties can be marked as internal and can also be accessed with dot notation.

    Example: obj = MyClassInheritingFromQualifiable(attributes={'id':1, 'custom':'ACustomValue'}, internal_names={'id'})
             print(obj.attributes['custom'])
             print(obj.attributes['id'])
             print(obj.id)
             obj.id = 2
             obj.attributes['id'] = 3
             obj.attributes['custom'] = 'AnotherCustomValue'
             obj.attributes['new_custom'] = 'newCustomValue'
    """
    def __init__(self, attributes: Dict[str, Any], internal_names: FrozenSet[str] = frozenset({})):
        # Setting attributes avoiding a call to the custom __setattr__()
        object.__setattr__(self, "_internal_names", ifnull(internal_names, frozenset({})))
        object.__setattr__(self, "_attributes", ifnull(attributes, {}))

    def encode(self):
        d = self.internal_attributes()

        d.update({
            "attributes": self.custom_attributes()
        })

        return d

    def __getattr__(self, name):
        if name in self.internal_names:
            return self._attributes.get(name, None)
        else:
            # Default behaviour
            raise AttributeError

    def __setattr__(self, key, value):
        if key in self.internal_names:
            self._attributes[key] = value
        else:
            # Default behaviour
            object.__setattr__(self, key, value)

    def get_attribute(self, name):
        """
        Get the value of an object attribute no matter it has been defined directly (with dot notation) or
        in the '_attributes' dictionary. It raises the AttributeError exception if not found.
        :param name: string with the name of the attribute
        :return: the value of the attribute
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if name in self._internal_names:
                return self._attributes.get(name, None)
            else:
                try:
                    return self._attributes[name]
                except KeyError:
                    raise AttributeError from None

    @property
    def internal_names(self) -> FrozenSet[str]:
        try:
            return object.__getattribute__(self, "_internal_names")
        except AttributeError:
            return frozenset({})

    @property
    def attributes(self) -> Dict[str, Any]:
        return self._attributes

    def custom_attributes(self) -> Dict[str, Any]:
        if self.attributes:
            return {k: v for k, v in self.attributes.items() if k not in self.internal_names}
        else:
            return {}

    def internal_attributes(self) -> Dict[str, Any]:
        if self.attributes:
            return {k: self.attributes.get(k, None) for k in self.internal_names}
        else:
            return {}

    def compare_attributes(self, attrs: Dict[str, Any]) -> bool:
        return all([attrs[k] == self.get_attribute(k) for k in attrs])


class Observable(Encodable):
    """ An entity which can be structurally (relations with other observables) or quantitatively observed.
        It can have none to infinite possible observations
    """
    def __init__(self):
        # self._location = location  # Definition of where the observable is
        self._physical_nature = None
        self._observations = []  # type: List[Union[FactorQuantitativeObservation, RelationObservation]]

    def encode(self):
        return {
            "observations": self.quantitative_observations
        }

    # Methods to manage the properties
    @property
    def observations(self):
        return self._observations

    @property
    def quantitative_observations(self) -> List["FactorQuantitativeObservation"]:
        return [o for o in self.observations if isinstance(o, FactorQuantitativeObservation)]

    def observations_append(self, observation):
        if isinstance(observation, (list, set)):
            lst = observation
        else:
            lst = [observation]

        self._observations.extend(lst)


class Geolocatable(Encodable):
    def __init__(self, geolocation: "Geolocation"):
        self._geolocation = geolocation

    def encode(self):
        return {
            "geolocation_ref": getattr(self.geolocation, "reference", None),
            "geolocation_code": getattr(self.geolocation, "code", None)
        }

    @property
    def geolocation(self):
        return self._geolocation


class HierarchyNode(Nameable, Encodable):
    """ Taxon, Processor and Factor
        A hierarchy node can be a member of several hierarchies ¿what's the name of this property?
        A hierarchy can be flat, so a hierarchy node can be member of a simple list
    """

    def __init__(self, name, parent=None, parent_weight=1.0, hierarchy=None, level=None, label=None, description=None, referred_node=None):
        Nameable.__init__(self, name)
        self._parents = []
        self._parents_weights = []
        self._referred_node: HierarchyNode = referred_node  # HierarchyNode in another Hierarchy
        self._children: Set[HierarchyNode] = set()  # HierarchyNode in the same "hierarchy"
        self._level = level  # type: HierarchyLevel
        self._hierarchy = hierarchy  # type: Hierarchy
        if referred_node and referred_node.hierarchy == self.hierarchy:
            raise Exception("The hierarchy of a node and the hierarchy of the referred cannot be the same")
        if hierarchy:  # Add name to the hierarchy
            hierarchy.codes[name] = self
            if not parent:
                hierarchy.roots_append(self)
        if parent:
            self.set_parent(parent, parent_weight)

        self._label = label
        self._description = description

    def encode(self):
        encoded_referred_node = None
        if self.referred_node:
            encoded_referred_node = {
                "name": self.referred_node.name,
                "hierarchy": getattr(self.referred_node.hierarchy, "name", None)
            }

        d = {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "parent": self.parent.name if self.parent else None,
            "referred_node": encoded_referred_node
        }
        return d

    @property
    def parent(self):
        """ Return the parent. It works when the node has only one parent """
        if len(self._parents) == 1:
            return self._parents[0]
            # return self._parents[next(iter(self._parents))]
        elif len(self._parents) == 0:
            return None
        else:
            raise Exception("The node has '" + str(len(self._parents)) + "' parents.")

    def set_parent(self, p: "HierarchyNode", weight=1.0):
        # Check that parent has the same type
        if p and type(p) is not self.__class__:
            raise Exception("The hierarchy node class is '" + str(self.__class__) +
                            "' while the type of the parent is '" + str(type(p)) + "'.")

        self._parents.append(p)
        if not weight:
            weight = 1.0
        self._parents_weights.append(weight)

        if p:
            p._children.add(self)

    def get_children(self):
        return self._children

    @staticmethod
    def hierarchically_related(p1: "HierarchyNode", p2: "HierarchyNode", h: "Hierarchy" = None):
        return p1.parent == p2 or p2.parent == p1

    @property
    def description(self):
        return self._description

    @property
    def label(self):
        return self._label

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level: "HierarchyLevel"):
        self._level = level

    @property
    def referred_node(self):
        return self._referred_node

    @property
    def hierarchy(self):
        return self._hierarchy

# #################################################################################################################### #
# VALUE OBJECTS for SPACE, TIME and QUALIFIED QUANTIFICATION
# #################################################################################################################### #


class Geolocation:
    """ For the geolocation of processors. Factors and Observations could also be qualified with Geolocation """
    def __init__(self, reference, code):
        self.reference = reference
        self.code = code

    def __eq__(self, other):
        return self.reference == other.reference and \
               self.code == other.code

    @staticmethod
    def create(reference, code) -> Optional["Geolocation"]:
        if reference and code:
            return Geolocation(reference, code)
        else:
            return None


class TimeExtent:
    """ For time location of Observations """
    def __init__(self, start, end=None):
        self._start = start
        self._end = end

    def __eq__(self, other):
        return self.start == other.start and \
               self.end == other.end

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._start = start

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, end):
        self._end = end

# #################################################################################################################### #
# HIERARCHIES and TAXA
# #################################################################################################################### #


class HierarchySource(Nameable, Identifiable):  # Organization defining the Hierarchy "vocabulary" and meanings
    def __init__(self, name):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)

    @staticmethod
    def partial_key(name: str=None, ident: str=None):
        d = dict(_t="hs")
        if name:
            d["_n"] = name
        if ident:
            d["__id"] = ident
        return d

    def key(self):
        return {"_t": "hs", "_n": self.name, "__id": self.ident}


# Also Hierarchical Code List (HCL). "Contains" one or more Hierarchy.
# If the HierarchyGroup has a single element named equally, it is a Code List
class HierarchyGroup(Nameable, Identifiable, Encodable):
    def __init__(self, name, source: HierarchySource=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        self._hierarchies = []  # A group can have several "Hierarchy"
        self._hierarchy_source = source

    def encode(self):
        return {
            "name": self.name,
            "source": self.hierarchy_source.name
        }

    @staticmethod
    def partial_key(name: str=None, ident: str=None):
        d = dict(_t="hg")
        if name:
            d["_n"] = name
        if ident:
            d["__id"] = ident
        return d

    def key(self):
        return {"_t": "hg", "_n": self.name, "__id": self.ident}

    @property
    def hierarchy_source(self):
        return self._hierarchy_source

    @property
    def hierarchies(self):
        return self._hierarchies


class HierarchyLevel(Nameable):  # Levels in a View. Code Lists do not have levels in SDMX, although it is allowed here
    def __init__(self, name, hierarchy):
        Nameable.__init__(self, name)
        self._hierarchy = hierarchy
        self._codes = set()  # HierarchyCodes in the level
        # hierarchy._level_names.append(name)
        # hierarchy._levels.append(hierarchy)

    @property
    def codes(self):
        return self._codes

    @property
    def hierarchy(self):
        return self._hierarchy


class Hierarchy(Nameable, Identifiable, Encodable):
    """
        A list or a forest of taxonomies (hierarchies), made of "Taxon" or "FactorType" instances
        (prepared also for "Processor" instances, not used)
    """
    def __init__(self, name: str=None, roots: List[HierarchyNode]=None, hierarchy_group: HierarchyGroup=None, type_name=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        # List of root "HierarchyNode" nodes (the object serves to represent a list)
        self._roots = []  # type: List[HierarchyNode]
        if roots:
            self.roots_append(roots)
        # The hierarchy should be of a single type
        # If not set here, the first element of the hierarchy sets the type, and new elements must be of the same type
        self._type = Hierarchy.__get_hierarchy_type(type_name)
        # Each level of the hierarchy can have a name. This list register these names, from root to leaves
        self._level_names = []  # type: List[str]

        # All HierarchyNodes contained by the Hierarchy. "code" to HierarchyNode
        self._codes = create_dictionary()  # type: Dict[HierarchyNode]
        self._hierarchy_group = hierarchy_group  # type: HierarchyGroup
        # List (ordered) of HierarchyLevels, from top to bottom
        self._levels = []  # type: List[HierarchyLevel]
        self._description = None

    def encode(self):
        def hierarchy_to_list(nodes: List[HierarchyNode]) -> List[HierarchyNode]:
            nodes_list = []
            if nodes:
                for node in sorted(nodes, key=lambda n: n.name):
                    nodes_list.append(node)
                    nodes_list.extend(hierarchy_to_list(list(node.get_children())))
            return nodes_list

        d = Encodable.parents_encode(self, __class__)

        d.update({
            "description": self._description,
            "hierarchy_group": None if not self.hierarchy_group else self.hierarchy_group.encode(),
            "nodes": hierarchy_to_list(self.roots)
        })

        return d

    @staticmethod
    def __get_hierarchy_type(type_name: Union[str, type]):
        ret = None
        if type_name:
            if isinstance(type_name, str):
                if type_name.lower() == "processor":
                    ret = Processor
                elif type_name.lower() in ["factortype", "interfacetype"]:
                    ret = FactorType
                elif type_name.lower() == "taxon":
                    ret = Taxon
            elif isinstance(type_name, type):
                if type_name in [Processor, FactorType, Taxon]:
                    ret = type_name
        else:
            ret = Taxon
        return ret

    @property
    def is_code_list(self):
        if self._roots:
            return self._roots[0].referred_node is None
        return False

    @property
    def roots(self):
        return self._roots

    def roots_append(self, root):
        if isinstance(root, (list, set)):
            if len(root) == 0:
                return
            first = root[0]
            lst = root
        else:
            first = root
            lst = [root]

        if not self._type:
            self._type = first

        self._roots.extend(lst)

    @property
    def hierarchy_type(self):
        return self._type

    @property
    def levels(self):
        return self._levels

    @property
    def level_names(self):
        return self._level_names

    @level_names.setter
    def level_names(self, lvls: list):
        self._level_names = lvls

    def get_node(self, name) -> HierarchyNode:
        """ Find a node of the hierarchy named "name" """
        def recursive_get_node(lst):
            for n in lst:
                if n.name.lower() == name.lower():
                    return n

            for n in lst:
                if n.get_children():
                    f = recursive_get_node(n.get_children())
                    if f:
                        return f
            return None

        return recursive_get_node(self._roots)

    @property
    def codes(self):
        return self._codes

    @property
    def hierarchy_group(self):
        return self._hierarchy_group

    @staticmethod
    def partial_key(name: str=None, hierarchy_type: Union[str, type]=None):
        d = dict(_t="h")
        if name:
            d["_n"] = name
        if hierarchy_type:
            d["_ht"] = Hierarchy.__get_hierarchy_type(hierarchy_type).__name__
        return d

    def key(self, alter_name=None):
        """
        Return a Key for the identification of the Hierarchy in the registry

        :param alter_name: Alternative name for the Hierarchy (multiple names are allowed for objects)
        :return:
        """
        return {"_t": "h",
                "_ht": self.hierarchy_type.__name__,
                "_n": self.name if not alter_name else alter_name,
                "__id": self.ident}

    def get_all_nodes(self):
        """
        Starting with roots, obtain a collection of all members of the hierarchy

        :return:
        """
        def recurse_node(n):
                if n.ident not in d:
                    if n._label:
                        d[n.ident] = (n.name, n._label)
                    else:
                        d[n.ident] = (n.name, n._description)
                    for c in n.get_children():
                        recurse_node(c)

        d = OrderedDict()
        for r in self.roots:
            if isinstance(r, Taxon):
                recurse_node(r)
        return d

    #####

    def get_codes(self):
        return self.to_dict().items()

    def to_dict(self):
        d = {}
        for c in self.codes.values():
            d[c._name] = c._description
        return d

    @staticmethod
    def construct_from_dict(d):
        return Hierarchy.construct(d["name"], d["description"], d["levels"], [CodeImmutable(**i) for i in d["codes"].values])

    @staticmethod
    def construct(name: str, description: str, levels: List[str], codes: List[CodeImmutable]):
        """

        :param name: Name of the Hierarchy
        :param description: Description of the Hierarchy
        :param levels: Names of the levels
        :param codes: List of codes, including in each the following tuple: CodeImmutable = namedtuple("CodeTuple", "code description level children")
        :return:
        """

        h = Hierarchy(name, roots=None)
        h._description = description
        # Levels
        levels_dict = create_dictionary()
        for l in levels:
            hl = HierarchyLevel(l, h)
            h._level_names.append(l)
            h._levels.append(hl)
            levels_dict[l] = hl
        # Codes
        codes_dict = create_dictionary()
        for ct in codes:
            hn = Taxon(ct.code, hierarchy=h, label=ct.description, description=ct.description)
            h.codes[ct.code] = hn
            hn.level = levels_dict.get(ct.level, None)  # Point to the containing HierarchyLevel
            if hn.level:
                hn.level._codes.add(hn)
            codes_dict[ct.code] = hn

        # Set children & parents
        for ct in codes:
            for ch in ct.children:
                if ch in codes_dict:
                    hn._children.add(codes_dict[ch])
                    codes_dict[ch]._parents.append(hn)
                    codes_dict[ch]._parents_weights.append(1.0)

        return h


class HierarchyExpression(Encodable):
    def __init__(self, expression: float=None):
        # Defines how to compute a HierarchyNode relative to other HierarchyNodes
        # The hierarchical relation implies a parent = SUM children expression. If specified, this implicit relation
        # would be overriden
        self._expression = expression

    def encode(self):
        return {
            "expression": self.expression
        }

    @property
    def expression(self):
        # Get expression defining
        return self._expression

    @expression.setter
    def expression(self, e: float=None):
        self._expression = e


class Taxon(Identifiable, HierarchyNode, HierarchyExpression, Qualifiable):
    """ For categories in a taxonomy. A taxonomy  """
    def __init__(self, name, parent=None, bottom_up_split=1.0, hierarchy=None, level=None, referred_taxon=None, expression=None, label=None, description=None, attributes=None):
        Identifiable.__init__(self)
        HierarchyNode.__init__(self, name, parent, bottom_up_split, hierarchy=hierarchy, level=level, referred_node=referred_taxon,
                               label=label, description=description)
        HierarchyExpression.__init__(self, expression)
        Qualifiable.__init__(self, attributes)
        self._description = description

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            "description": self.description
        })

        return d

    @staticmethod
    def partial_key(name: str=None, ident: str=None):
        d = dict(_t="t")
        if name:
            d["_n"] = name
        if ident:
            d["__id"] = ident
        return d

    def key(self):
        return {"_t": "t", "_n": self.name, "__id": self.ident}


# #################################################################################################################### #
# CONTEXTS and REFERENCES. Both are to contain dictionaries of Attributes.

# References restrict possible attributes depending on the profile: bibliography, geography
# #################################################################################################################### #

class Context(Identifiable, Nameable, Qualifiable):
    """
    A context is just a named container for attribute sets (Qualifiable)

    Parameters, ScaleChangers, ProcessorScalings, ETL or dataset transforms -and other possible adaptive
    MuSIASEM elements-, can take into account how the attributes of base MuSIASEM elements: Processors, Interfaces,
    Hierarchies of Categories, InterfaceTypes, MATCH before applying specific coefficients.
    """
    def __init__(self, name, attributes):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        Qualifiable.__init__(self, attributes)

    @staticmethod
    def partial_key(name: str=None, ident: str=None):
        d = dict(_t="ctx")
        if name:
            d["_n"] = name
        if ident:
            d["__id"] = ident
        return d

    def key(self):
        return {"_t": "ctx", "_n": self.name, "__id": self.ident}


class Observer(Identifiable, Nameable, Encodable):
    """ An entity capable of producing Observations on an Observable
        In our context, it is a process obtaining data
    """
    no_observer_specified = "_default_observer"

    def __init__(self, name, description=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        self._description = description  # Informal description of the observation process (it could be just a manual transcription of numbers from a book)
        self._observation_process_description = None  # Formal description of the observation process
        self._observables = []  # type: List[Observable]

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            "description": self._description,
            "observation_process_description": self._observation_process_description,
            "observables": [{"name": obs.name, "id": obs.uuid} for obs in self.observables if obs]
        })

        return d

    @property
    def observables(self):
        return self._observables

    def observables_append(self, oble):
        if isinstance(oble, (list, set)):
            lst = oble
        else:
            lst = [oble]

        for o in lst:
            if o not in self._observables:
                self._observables.append(o)

    @staticmethod
    def partial_key(name: str=None):
        if name:
            return {"_t": "o", "_n": name}
        else:
            return {"_t": "o"}

    def key(self):
        """
        Return a Key for the identification of the Observer in the registry
        :param registry:
        :return:
        """
        return {"_t": "o", "_n": self.name, "__id": self.ident}


class Reference(Identifiable, Nameable, Qualifiable):
    """ A dictionary containing a set of key-value pairs
        with some validation schema (search code for "ref_prof" global variable)
    """
    def __init__(self, name, content):
        """
        :param name:
        :param content:
        """
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        Qualifiable.__init__(self, content)


class GeographicReference(Reference, Encodable):
    def __init__(self, name, content):
        """
        :param name:
        :param content:
        """
        Reference.__init__(self, name, content)

    # TODO A method to validate that attributes

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        return d

    @staticmethod
    def partial_key(name: str=None):
        d = dict(_t="rg")
        if name:
            d["_n"] = name
        return d

    def key(self):
        return {"_t": "rg", "_n": self.name, "__o": self.ident}


class BibliographicReference(Reference, Observer, Encodable):
    def __init__(self, name, content):
        """
        :param name:
        :param content:
        """
        Reference.__init__(self, name, content)
        Observer.__init__(self, name, description=None)

    # TODO A method to validate that attributes

    def encode(self):
        d = {"type": "bibliographic_reference"}

        d.update(Encodable.parents_encode(self, __class__))

        return d

    @staticmethod
    def partial_key(name: str=None):
        d = dict(_t="rb")
        if name:
            d["_n"] = name
        return d

    def key(self):
        return {"_t": "rb", "_n": self.name, "__o": self.ident}


class ProvenanceReference(Reference, Observer, Encodable):
    def __init__(self, name, content):
        """
        :param name:
        :param content:
        """
        Reference.__init__(self, name, content)
        Observer.__init__(self, name, description=None)

    # TODO A method to validate that attributes

    def encode(self):
        d = {"type": "provenance_reference"}

        d.update(Encodable.parents_encode(self, __class__))

        return d

    @staticmethod
    def partial_key(name: str=None):
        d = dict(_t="rp")
        if name:
            d["_n"] = name
        return d

    def key(self):
        return {"_t": "rp", "_n": self.name, "__o": self.ident}


# #################################################################################################################### #
# Entities
# #################################################################################################################### #


class Source(Nameable, Qualifiable):
    """ Any source of information, used by an Observer to elaborate Observations. It can be an externally obtained
        Dataset or other
    """
    def __init__(self, attributes):
        attributes2 = attributes.clone()
        # TODO If present, remove the following attributes from the dictionary
        self._description = None  # What is the content of the source
        self._location = None  # Where is the source accessible
        self._address = None  # Address to have direct access to the source
        Qualifiable.__init__(self, attributes2)


class FactorType(Identifiable, HierarchyNode, HierarchyExpression, Taggable, Qualifiable, Encodable):
    """ A Factor as type, in a hierarchy, a Taxonomy """
    INTERNAL_ATTRIBUTE_NAMES = frozenset({})

    def __init__(self, name, parent=None, hierarchy=None, roegen_type: FlowFundRoegenType=FlowFundRoegenType.flow,
                 tags=None, attributes=None, expression=None, sphere=None, opposite_processor_type=None):
        Identifiable.__init__(self)
        HierarchyNode.__init__(self, name, parent, hierarchy=hierarchy)
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes, self.INTERNAL_ATTRIBUTE_NAMES)
        HierarchyExpression.__init__(self, expression)
        self._roegen_type = roegen_type
        self._sphere = sphere
        self._opposite_processor_type = opposite_processor_type
        self._physical_type = None  # TODO Which physical types. An object
        self._default_unit_str = None  # TODO A string representing the unit, compatible with the physical type
        self._factors = []

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            'roegen_type': getattr(self.roegen_type, "name", None),
            'sphere': self.sphere,
            'opposite_processor_type': self.opposite_processor_type
        })

        # Remove property inherited from HierarchyNode because it is always "null" for FactorType
        d.pop("referred_node", None)

        return d

    def full_hierarchy_name(self):
        """
        Obtain the full hierarchy name of the current FactorType

        :return:
        """
        p = self
        lst = []
        while p:
            lst.insert(0, p.name)
            par = getattr(p, "parent", None)
            if par:
                p = p.parent
            else:
                p = None
        return ".".join(lst)

    # A registry of all factors referring to this FactorType
    @property
    def factors(self):
        return self._factors

    def factors_append(self, factor: "Factor"):
        self._factors.append(factor)

    @property
    def roegen_type(self):
        return self._roegen_type

    @property
    def opposite_processor_type(self):
        return self._opposite_processor_type

    @property
    def sphere(self):
        return self._sphere

    def simple_name(self):
        parts = self.name.split(".")
        if len(parts) > 1:
            return parts[-1]
        else:
            return self.name

    @staticmethod
    def alias_key(name: str, factor_type: "FactorType"):
        """
        Creates a registrable entry to allow a different name for "factor_type"
        Key TO Key
        :param name:
        :param factor_type:
        :return: A tuple formed by the key and value to be registered
        """
        return {"_t": "ft", "_n": name, "__id": factor_type.ident, "_aka": True}

    @staticmethod
    def partial_key(name: str=None, ident: str=None):
        d = dict(_t="ft")
        if name:
            d["_n"] = name
        if ident:
            d["__id"] = ident
        return d

    def key(self):
        # Name here is not a unique identifier. The primary key is the "__id" and "_t" helps to reduce search options
        # when retrieving factor types
        return {"_t": "ft", "_n": self.name, "__id": self.ident}


class Processor(Identifiable, Nameable, Taggable, Qualifiable, Automatable, Observable, Geolocatable, Encodable):
    INTERNAL_ATTRIBUTE_NAMES = frozenset({
        'subsystem_type', 'processor_system', 'functional_or_structural', 'instance_or_archetype', 'stock'
    })

    def __init__(self, name, attributes: Dict[str, Any] = None, geolocation: Geolocation = None, tags=None,
                 referenced_processor: "Processor" = None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes, self.INTERNAL_ATTRIBUTE_NAMES)
        Automatable.__init__(self)
        Observable.__init__(self)
        Geolocatable.__init__(self, geolocation)

        self._factors = []  # type: List[Factor]
        self._relationships = []  # type: List[ProcessorsRelationObservation]
        self._local_indicators = []  # type: List[Indicator]

        # The processor references a previously defined processor
        # * If a factor is defined, do not look in the referenced
        # * If a factor is not defined here, look in the referenced
        # * Other properties can also be referenced (if defined locally it is not searched in the referenced)
        # * FactorObservations are also assimilated
        #   (although expressions evaluated in the context of the local processor, not the referenced one)
        # * Relations are NOT assimilated, at least by default. THIS HAS TO BE STUDIED IN DEPTH
        self._referenced_processor = referenced_processor

    def encode(self):
        d = Identifiable.encode(self)
        d.update(Nameable.encode(self))
        d.update(Qualifiable.encode(self))
        d.update(Geolocatable.encode(self))
        d.update(Automatable.encode(self))

        d.update({
            'interfaces': self.factors
        })

        return d

    @property
    def referenced_processor(self):
        return self._referenced_processor

    @property
    def factors(self):
        tmp = [f for f in self._factors]
        if self.referenced_processor:
            s = set([f.name.lower() for f in tmp])
            for f in self.referenced_processor.factors:
                if f.name.lower() not in s:
                    tmp.append(f)
                    s.add(f.name().lower())
        return tmp

    def factors_append(self, factor: "Factor"):
        self._factors.append(factor)

    def factors_find(self, factor_name: str) -> Optional["Factor"]:
        for f in self.factors:  # type: Factor
            if strcmp(f.name, factor_name):
                return f

        return None

    @property
    def extensive(self):
        # TODO True if of all values defined for all factors no value depends on another factor
        return False

    @property
    def intensive(self):
        return not self.extensive

    def simple_name(self):
        parts = self.name.split(".")
        if len(parts) > 1:
            return parts[-1]
        else:
            return self.name

    def full_hierarchy_names(self, registry: PartialRetrievalDictionary) -> List[str]:
        """
        Obtain the full hierarchy name of the current processor
        It looks for the PART-OF relations in which the processor is in the child side
        It can return multiple names because the same processor can be child of different processors

        :param registry:
        :return:
        """
        # Get matching parent relations
        parent_relations = registry.get(ProcessorsRelationPartOfObservation.partial_key(child=self))

        # Compose the name, recursively
        if len(parent_relations) == 0:
            return [self.name]
        else:
            # Take last part of the name
            # last_part = self.name.split(".")[-1]
            return [(parent_name+"."+self.name)
                    for parent_relation in parent_relations
                    for parent_name in parent_relation.parent_processor.full_hierarchy_names(registry)]

    def children(self, registry: PartialRetrievalDictionary) -> List["Processor"]:
        """
        Obtain the list of children looking for the PART-OF relations

        :param registry:
        :return:
        """
        return [r.child_processor for r in registry.get(ProcessorsRelationPartOfObservation.partial_key(parent=self))]

    def clone(self, state: Union[PartialRetrievalDictionary, State], objects_already_cloned: Dict = None, level=0,
              inherited_attributes: Dict[str, Any] = {}, name: str = None):
        """
        Processor elements:
         - Attributes. Generic; type, external, stock
         - Location
         - Factors
           - Observations
           - Relations
         - Local_indicators
         - Relations:
           - part-of: if the processor is the parent part, clone recursively (include the part-of relations)
           - undirected flow

        = Reference cloned processor (or not)
        = Register in Processor Set (or not)

        :param state: Global state
        :param objects_already_cloned: Dictionary containing already cloned Processors and Factors (INTERNAL USE)
        :param level: Recursion level (INTERNAL USE)
        :param inherited_attributes: Attributes for the new Processor
        :param name: Name for the new Processor (if None, adopt the name of the cloned Processor)
        :return: cloned processor, cloned children
        """
        if isinstance(state, PartialRetrievalDictionary):
            glb_idx = state
        else:
            glb_idx, _, _, _, _ = get_case_study_registry_objects(state)

        # Create new Processor
        if not name:
            name = self.name
        p = Processor(name,
                      attributes={**self.attributes, **inherited_attributes},
                      geolocation=self.geolocation,
                      tags=self._tags,
                      referenced_processor=self.referenced_processor
                      )

        # if name != self.name:
        #     glb_idx.put(p.key(), p)

        if not objects_already_cloned:
            objects_already_cloned = {}

        objects_already_cloned[self] = p

        # Factors
        for f in self.factors:
            f_ = Factor.clone_and_append(f, p)  # The Factor is cloned and appended into the Processor "p"
            glb_idx.put(f_.key(), f_)
            if f not in objects_already_cloned:
                objects_already_cloned[f] = f_
            else:
                raise Exception("Unexpected: the factor "+f.processor.name+":"+f.taxon.name+" should not be cloned again.")

        # Local indicators
        for li in self._local_indicators:
            formula = li._formula  # TODO Adapt formula of the new indicator to the new Factors
            li_ = Indicator(li._name, formula, li, li._benchmark, li._indicator_category)
            self._local_indicators.append(li_)

        # Clone child Processors (look for part-of relations)
        children_processors: Set[Processor] = set()
        part_of_relations = glb_idx.get(ProcessorsRelationPartOfObservation.partial_key(parent=self))
        for rel in part_of_relations:
            if rel.child_processor not in objects_already_cloned:
                p2, p2_children = rel.child_processor.clone(state, objects_already_cloned, level + 1, inherited_attributes)  # Recursive call
                objects_already_cloned[rel.child_processor] = p2
                children_processors.add(p2)
                children_processors |= p2_children
            else:
                p2 = objects_already_cloned[rel.child_processor]

            # Clone part-of relation
            o1 = ProcessorsRelationPartOfObservation.create_and_append(p, p2, rel.observer)
            glb_idx.put(o1.key(), o1)

            # If there is a Upscale relation, clone it also
            upscale_relations = glb_idx.get(ProcessorsRelationUpscaleObservation.partial_key(parent=self, child=rel.child_processor))
            if upscale_relations:
                factor_name = upscale_relations[0].factor_name
                quantity = upscale_relations[0].quantity
                o2 = ProcessorsRelationUpscaleObservation.create_and_append(p, p2, rel.observer, factor_name, quantity)
                glb_idx.put(o2.key(), o2)

        # PROCESS FLOW relations: directed and undirected (if at entry level, i.e., level == 0)
        # This step is needed because flow relations may involve processors and flows outside of the clone processor.
        # If a flow is totally contained, pointing to two cloned elements, the cloned flow will point to the two new elements.
        # If a flow points to an element out, the cloned flow will point have at one side a new element, at the other an existing one.
        if level == 0:
            considered_flows = set()  # Set of already processed flows
            considered_scales = set()  # Set of already processed scales
            for o in objects_already_cloned:
                if isinstance(o, Factor):
                    # Flows where the Interface is origin
                    for f in glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key(source=o)):
                        if f not in considered_flows:
                            new_f = FactorsRelationDirectedFlowObservation(source=objects_already_cloned[o],
                                                                           target=objects_already_cloned.get(f.target_factor, f.target_factor),
                                                                           observer=f.observer,
                                                                           weight=f.weight,
                                                                           tags=f.tags,
                                                                           attributes=f.attributes,
                                                                           back=objects_already_cloned.get(f.back_factor, f.back_factor))
                            glb_idx.put(new_f.key(), new_f)
                            considered_flows.add(f)
                    # Flows where the Interface is destination
                    for f in glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key(target=o)):
                        if f not in considered_flows:
                            new_f = FactorsRelationDirectedFlowObservation(source=objects_already_cloned.get(f.source_factor, f.source_factor),
                                                                           target=objects_already_cloned[o],
                                                                           observer=f.observer,
                                                                           weight=f.weight,
                                                                           tags=f.tags,
                                                                           attributes=f.attributes,
                                                                           back=objects_already_cloned.get(f.back_factor, f.back_factor))

                            glb_idx.put(new_f.key(), new_f)
                            considered_flows.add(f)
                    # Flows where the Interface is back interface
                    for f in glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key(back=o)):
                        if f not in considered_flows:
                            new_f = FactorsRelationDirectedFlowObservation(source=objects_already_cloned.get(f.source_factor, f.source_factor),
                                                                           target=objects_already_cloned.get(f.target_factor, f.target_factor),
                                                                           observer=f.observer,
                                                                           weight=f.weight,
                                                                           tags=f.tags,
                                                                           attributes=f.attributes,
                                                                           back=objects_already_cloned[o])

                            glb_idx.put(new_f.key(), new_f)
                            considered_flows.add(f)
                    # Scales
                    for f in glb_idx.get(FactorsRelationScaleObservation.partial_key(origin=o)):
                        if f not in considered_scales:
                            new_f = FactorsRelationScaleObservation(origin=objects_already_cloned[o],
                                                                    destination=objects_already_cloned.get(f.destination, f.destination),
                                                                    observer=f.observer,
                                                                    quantity=f.quantity,
                                                                    tags=f.tags,
                                                                    attributes=f.attributes)
                            glb_idx.put(new_f.key(), new_f)
                            considered_scales.add(f)
                    # Scales where the Interface is destination
                    for f in glb_idx.get(FactorsRelationScaleObservation.partial_key(destination=o)):  # Destination
                        if f not in considered_scales:
                            new_f = FactorsRelationScaleObservation(
                                origin=objects_already_cloned.get(f.origin, f.origin), destination=objects_already_cloned[o],
                                observer=f.observer, quantity=f.quantity, tags=f.tags, attributes=f.attributes)

                            glb_idx.put(new_f.key(), new_f)
                            considered_scales.add(f)

                else:  # "o" is a Processor
                    for f in glb_idx.get(ProcessorsRelationUndirectedFlowObservation.partial_key(source=o)): # As Source
                        if f not in considered_flows:
                            new_f = ProcessorsRelationUndirectedFlowObservation(source=o, target=objects_already_cloned.get(f.target_factor, f.target_factor), observer=f.observer, weight=f.weight, tags=f.tags, attributes=f.attributes)
                            glb_idx.put(new_f.key(), new_f)
                            considered_flows.add(f)
                    for f in glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key(target=o)): # As Target
                        if f not in considered_flows:
                            new_f = ProcessorsRelationUndirectedFlowObservation(source=objects_already_cloned.get(f.source_factor, f.source_factor), target=o, observer=f.observer, weight=f.weight, tags=f.tags, attributes=f.attributes)

                            glb_idx.put(new_f.key(), new_f)
                            considered_flows.add(f)

        return p, children_processors

    @staticmethod
    def register(processors: List["Processor"], registry: PartialRetrievalDictionary):
        """Add processors' hierarchical names to global register"""
        for processor in processors:
            for hierarchical_name in processor.full_hierarchy_names(registry):
                registry.put(Processor.partial_key(name=hierarchical_name, ident=processor.ident), processor)

    @staticmethod
    def alias_key(name: str, processor: "Processor"):
        """
        Creates a registrable entry to allow a different name for "processor"
        Key TO Key
        :param name:
        :param processor:
        :return: A tuple formed by the key and value to be registered
        """
        return {"_t": "p", "_n": name, "__id": processor.ident, "_aka": True}

    @staticmethod
    def is_alias_key(composite_key: dict):
        return "_aka" in composite_key

    @staticmethod
    def partial_key(name: str=None, ident: str=None):
        d = dict(_t="p")
        if name:
            d["_n"] = name
        if ident:
            d["__id"] = ident
        return d

    def key(self):
        """
        Return a Key for the identification of the Processor in the registry
        :return:
        """
        # Name here is not a unique identifier. The primary key is the "__id" and "_t" helps to reduce search options
        # when retrieving processors
        return {"_t": "p", "_n": self.name, "__id": self.ident}


class Factor(Identifiable, Nameable, Taggable, Qualifiable, Observable, Automatable, Geolocatable, Encodable):
    """ A Flow or Fund, when attached to a Processor
        It is automatable because an algorithm emulating an expert could inject Factors into Processors (as well as
        associated Observations)
    """
    INTERNAL_ATTRIBUTE_NAMES = frozenset({
        'sphere', 'roegen_type', 'orientation', 'opposite_processor_type'
    })

    def __init__(self, name, processor: Processor, in_processor_type: FactorInProcessorType, taxon: FactorType=None,
                 referenced_factor: "Factor" = None,
                 geolocation: Geolocation = None, tags=None, attributes=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes, self.INTERNAL_ATTRIBUTE_NAMES)
        Observable.__init__(self)
        Automatable.__init__(self)
        Geolocatable.__init__(self, geolocation)

        self._processor = processor
        self._taxon = taxon
        self._type = in_processor_type

        # The factor references a previously defined factor
        # * It can contain its own observations
        # * Inherits all the None properties
        # * Relations are NOT assimilated, at least by default. THIS HAS TO BE STUDIED IN DEPTH
        self._referenced_factor = referenced_factor

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            'type': {"external": self.type.external, "incoming": self.type.incoming} if self.type else None,
            'interface_type': name_and_id_dict(self.taxon),
        })

        return d

    @property
    def full_name(self):
        return self.processor.name + ":" + self.name

    @property
    def processor(self):
        return self._processor

    @property
    def referenced_factor(self):
        return self._referenced_factor

    @staticmethod
    def create_and_append(name, processor: Processor, in_processor_type: FactorInProcessorType, taxon: FactorType,
                          geolocation: Geolocation = None, tags=None, attributes=None):
        f = Factor(name=name, processor=processor, in_processor_type=in_processor_type, taxon=taxon,
                   geolocation=geolocation, tags=tags, attributes=attributes)
        if processor:
            processor.factors_append(f)
        if taxon:
            taxon.factors_append(f)
        return f

    @property
    def type(self):  # External (or internal), Incoming (or outgoing)
        tmp = self._type
        if tmp is None and self.referenced_factor:
            tmp = self.referenced_factor.type

        return tmp

    @property
    def taxon(self) -> FactorType:
        tmp = self._taxon
        if tmp is None and self.referenced_factor:
            tmp = self.referenced_factor.taxon

        return tmp

    # @property
    # def roegen_type(self):
    #     tmp = self._taxon.roegen_type
    #     if tmp is None and self.referenced_factor:
    #         tmp = self.referenced_factor.roegen_type
    #
    #     return tmp

    # Overwritten. Because the observations of the factor can be the observations from a referenced processor
    @property
    def observations(self):
        tmp = [o for o in self._observations]
        if self.referenced_factor:
            for o in self.referenced_factor.observations:
                tmp.append(o)
        return tmp

    @staticmethod
    def clone_and_append(factor: "Factor", processor: Processor):
        f = Factor.create_and_append(factor.name, processor, in_processor_type=factor.type, taxon=factor._taxon,
                                     geolocation=factor.geolocation, tags=factor._tags, attributes=factor.attributes)
        # Observations (only Quantities, because they are owned by the Factor)
        for o in factor.observations:
            if isinstance(o, FactorQuantitativeObservation):
                FactorQuantitativeObservation.create_and_append(o.value, f, o.observer, o.tags, o.attributes)

        return f

    @staticmethod
    def partial_key(processor: Processor=None, factor_type: FactorType=None, name: str=None):
        d = {"_t": "f"}
        if processor:
            d["__p"] = processor.ident
        if factor_type:
            d["__ft"] = factor_type.ident
        if name:
            d["_n"] = name

        return d

    def key(self):
        """
        Return a Key for the identification of the Factor in the registry
        :return:
        """
        return {"_t": "f", "__id": self.ident, "__p": self._processor.ident, "__ft": self._taxon.ident, "_n": self.name}


class ProcessorsSet(Nameable):
    """ A set of Processors """
    def __init__(self, name):
        Nameable.__init__(self, name)
        self._pl = set([])  # Processors members of the set
        self._attributes = create_dictionary()  # Attributes characterizing the processors. {attribute_name: code list}

    def append(self, p: Processor, prd: PartialRetrievalDictionary):
        """
        Append a Processor, but ONLY if it does not exist
        Existence is determined by: name, taxa, position

        :param p:
        :return:
        """
        res = self.search(p.full_hierarchy_names(prd), p.attributes, prd)
        if not res:
            self._pl.add(p)
            return p
        else:
            if len(res) > 1:
                raise Exception("The match should not involve more than 1 processor (0 or 1 allowed)")
            else:
                return res[0]

    def append_attributes_codes(self, d: dict):
        """
        Receives a dictionary with attributes (key, value) of a processor
        with the goal of elaborating a code-list for each attribute

        :param d:
        :return:
        """
        for k in d:  # "k" is the name (the header) of the attribute, d[k] is one of the possible values
            if k not in self._attributes:
                s = set()
                self._attributes[k] = s
            else:
                s = self._attributes[k]
            s.add(d[k])

    @property
    def attributes(self):
        return self._attributes

    def search(self, names: List[str], attrs: dict, prd: PartialRetrievalDictionary) -> List[Processor]:
        def matches() -> bool:
            # Check if hierarchical names match
            s1 = set([i.lower() for i in names])
            s2 = set([i.lower() for i in p.full_hierarchy_names(prd)])
            if not s1.intersection(s2):  # Empty intersection -> different names -> Do not match
                return False
            elif p.attributes and attrs:  # If names are equal but equal attributes -> Match
                if case_sensitive:
                    s1 = set(attrs.items())
                    s2 = set(p.attributes.items())
                else:
                    s1 = set((k.lower(), v.lower() if v else None) for k, v in attrs.items())
                    s2 = set((k.lower(), v.lower() if v else None) for k, v in p.attributes.items())
                return not s1.difference(s2)  # If no difference in attributes -> Match
            else:
                return False

        lst = []
        for p in self._pl:
            if matches():
                lst.append(p)
        return lst

    def clone(self):
        pass

    @staticmethod
    def partial_key(name: str=None):
        d = dict(_t="ps")
        if name:
            d["_n"] = name
        return d

    def key(self):
        return {"_t": "ps", "_n": self.name}


####################################################################################################
# NOT USABLE RIGHT NOW!!!!!!!!!!!!!!!!!!!!!!!!!
####################################################################################################
# def connect_processors(source_p: Processor, dest_p: Processor, h: "Hierarchy", weight: float, taxon: FactorType, source_name: str=None, dest_name: str=None):
#     """ High level function to connect two processors adding factors if necessary """
#     if not dest_name and source_name:
#         dest_name = source_name
#     hierarchical = HierarchyNode.hierarchically_related(source_p, dest_p, h)
#     f1 = None
#     f2 = None
#     if source_name:
#         # Find a factor with source_name in source_p
#         for f in source_p.factors:
#             if source_name.lower() == f.name:
#                 f1 = f
#                 break
#         # If found, the type should be congruent. If it is an hierarchical connection, the sense is
#         # reversed (IN allows a connection OUT to its children, which must be also IN)
#         if f1:
#             # TODO Check all possibilities with tests (in hierarchical, out hierarchical, in sequential, out sequential)
#             ok = (not f1.type.incoming and not hierarchical) or (f1.type.incoming and hierarchical)
#             if not ok:
#                 raise Exception("A factor by the name '"+source_name+"' already exists for source Processor "
#                                 "'"+source_p.name+"' and the sense of connection is not congruent with connection "
#                                 "to be added")
#     if dest_name:
#         # Find a factor with dest_name in dest_p
#         for f in dest_p.factors:
#             if dest_name.lower() == f.name:
#                 f2 = f
#                 break
#         # If found, the type should be congruent. If it is an hierarchical connection, the sense is
#         # reversed (IN allows a connection OUT to its children, which must be also IN)
#         if f2:
#             # TODO Check all possibilities with tests (in hierarchical, out hierarchical, in sequential, out sequential)
#             ok = (f2.type.incoming and not hierarchical) or (not f2.type.incoming and hierarchical)
#             if not ok:
#                 raise Exception("A factor by the name '"+source_name+"' already exists for destination Processor "
#                                 "'"+dest_p.name+"' and the sense of connection is not congruent with connection "
#                                 "to be added")
#
#     if not f1:
#         if not hierarchical:
#             incoming = False
#         else:
#             incoming = True
#         f1 = Factor.create(source_name, source_p, FactorInProcessorType(external=source_p.external, incoming=incoming), taxon)
#
#     if not f2:
#         if not hierarchical:
#             incoming = True
#         else:
#             incoming = False
#         f2 = Factor.create(dest_name, dest_p, FactorInProcessorType(external=dest_p.external, incoming=incoming), taxon)
#
#     c = f1.connect_to(f2, h, weight)
#
#     return c, f1, f2

# #################################################################################################################### #
# <<<< OBSERVATION CLASSES >>>>
# #################################################################################################################### #


class RelationClassType(Enum):
    # Relations between Processors
    pp_part_of = ("part_of", "partof", "|")  # Left is inside Right. No assumptions on flows between child and parent.
    pp_undirected_flow = ("<>", "><")
    pp_upscale = "||"
    pp_isa = ("is_a", "isa")  # "Left" gets a copy of ALL "Right" interface types
    # pp_asa = ("as_a", "asa")  # Left must already have ALL interfaces from Right. Similar to "part-of" in the sense that ALL Right interfaces are connected from Left to Right
    # pp_compose = "compose"
    pp_aggregate = ("aggregate", "aggregation")
    pp_associate = ("associate", "association")

    # Relations between Interfaces
    ff_directed_flow = ("flow", ">")
    ff_reverse_directed_flow = "<"
    ff_scale = "scale"
    ff_directed_flow_back = ("flowback", "flow_back")
    ff_scale_change = ("scalechange", "scale_change")

    # Relations between Interface Types
    ftft_directed_linear_transform = "interface_type_change"

    def __init__(self, *labels):
        self.labels = labels

    @property
    def is_between_processors(self) -> bool:
        return self.name.startswith("pp_")

    @property
    def is_between_interfaces(self) -> bool:
        return self.name.startswith("ff_")

    @property
    def is_flow(self) -> bool:
        return self.name.endswith("_flow")

    @staticmethod
    def from_str(label: str) -> "RelationClassType":

        for member in list(RelationClassType):
            if label.lower() in member.labels:
                return member

        raise NotImplementedError(f"The relation type '{label}' is not valid.")

    @staticmethod
    def all_labels() -> List[str]:
        return [label for member in list(RelationClassType) for label in member.labels]


class FactorQuantitativeObservation(Taggable, Qualifiable, Automatable, Encodable):
    """ An expression or quantity assigned to an Observable (Factor) """
    def __init__(self, v: Union[str, float], factor: Factor=None, observer: Observer=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._value = v
        self._factor = factor
        self._observer = observer

    def encode(self):
        d = {
            "value": self.value
            #"interface": self.factor,
            #"observer": self.observer
        }
        return d

    @staticmethod
    def create_and_append(v: Union[str, float], factor: Factor, observer: Observer, tags=None, attributes=None):
        o = FactorQuantitativeObservation(v, factor, observer, tags, attributes)
        if factor:
            factor.observations_append(o)
        if observer:
            observer.observables_append(factor)
        return o

    @property
    def factor(self):
        return self._factor

    @property
    def observer(self):
        return self._observer

    @property
    def value(self):
        return self._value

    @property
    def is_relative(self):
        return (self.attributes.get("relative_to") is not None) if self.attributes else False

    @property
    def relative_factor(self):
        return self.attributes.get("relative_to") if self.attributes else None

    @value.setter
    def value(self, v):
        self._value = v

    @staticmethod
    def partial_key(factor: Factor = None, observer: Observer = None, relative: bool = None):
        d = {"_t": "qq"}
        if relative:
            d["__rt"] = relative

        if factor:
            d["__f"] = factor.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self):
        d = {"_t": "qq", "__f": self._factor.ident, "__rt": self.is_relative}
        if self._observer:
            d["__oer"] = self._observer.ident
        return d


class RelationObservation(Taggable, Qualifiable, Automatable, Encodable):  # All relation kinds

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        return d


class ProcessorsRelationObservation(RelationObservation, Encodable):  # Just base of ProcessorRelations

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        return d


class FactorTypesRelationObservation(RelationObservation, Encodable):  # Just base of FactorTypesRelations

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        return d


class FactorsRelationObservation(RelationObservation, Encodable):  # Just base of FactorRelations

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        return d


class FactorTypesRelationUnidirectionalLinearTransformObservation(FactorTypesRelationObservation, Encodable):
    """
    Expression of an Unidirectional Linear Transform, from an origin FactorType to a destination FactorType
    A weight can be a expression containing parameters
    This relation will be applied to Factors which are instances of the origin FactorTypes, to obtain destination
    FactorTypes
    """
    def __init__(self, origin: FactorType, destination: FactorType, weight: Union[float, str],
                 origin_context: Processor = None, destination_context: Processor = None, origin_unit=None,
                 destination_unit=None, observer: Observer = None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)

        self._origin = origin
        self._destination = destination
        self._weight = weight
        self._observer = observer
        self._origin_context = origin_context
        self._destination_context = destination_context
        self._origin_unit = origin_unit
        self._destination_unit = destination_unit

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            "origin": name_and_id_dict(self.origin),
            "destination": name_and_id_dict(self.destination),
            "weight": self._weight,
            "observer": name_and_id_dict(self.observer),
            "origin_context": self._origin_context,
            "destination_context": self._destination_context,
            "origin_unit": self._origin_unit,
            "destination_unit": self._destination_unit
        })

        return d

    def __str__(self):
        return f"ScaleChange: from {self._origin.name} to {self._destination.name}, origin ctx {self._origin_context.name if self._origin_context else '-'}, dst ctx {self._destination_context.name if self._destination_context else '-'}"

    @staticmethod
    def create_and_append(origin: FactorType, destination: FactorType, weight, origin_context: Processor = None,
                          destination_context: Processor = None, origin_unit=None, destination_unit=None,
                          observer: Observer = None, tags=None, attributes=None):
        o = FactorTypesRelationUnidirectionalLinearTransformObservation(
            origin, destination, weight, origin_context, destination_context,
            origin_unit, destination_unit, observer, tags, attributes)

        # if origin:
        #     origin.observations_append(o)
        # if destination:
        #     destination.observations_append(o)
        if observer:
            observer.observables_append(origin)
            observer.observables_append(destination)
        return o

    @property
    def origin(self):
        return self._origin

    @property
    def destination(self):
        return self._destination

    @property
    def scaled_weight(self):
        return UnitConversion.get_scaled_weight(self._weight,
                                                self.origin.attributes.get("unit"), self._origin_unit,
                                                self._destination_unit, self.destination.attributes.get("unit"))

    @property
    def observer(self):
        return self._observer

    @staticmethod
    def partial_key(origin: FactorType = None, destination: FactorType = None,
                    origin_context: Processor = None, destination_context: Processor = None, observer: Observer = None):
        d = {"_t": RelationClassType.ftft_directed_linear_transform.name}
        if origin:
            d["__o"] = origin.ident

        if destination:
            d["__d"] = destination.ident

        if observer:
            d["__oer"] = observer.ident

        if origin_context:
            d["__oc"] = origin_context.ident

        if destination_context:
            d["__dc"] = destination_context.ident

        return d

    def key(self):
        d = {"_t": RelationClassType.ftft_directed_linear_transform.name,
             "__o": self._origin.ident, "__d": self._destination.ident}

        if self._origin_context:
            d["__oc"] = self._origin_context.ident

        if self._destination_context:
            d["__dc"] = self._destination_context.ident

        if self._observer:
            d["__oer"] = self._observer.ident
        return d


class ProcessorsRelationIsAObservation(ProcessorsRelationObservation, Encodable):
    def __init__(self, parent: Processor, child: Processor, observer: Observer=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._parent = parent
        self._child = child
        self._observer = observer

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            "origin": name_and_id_dict(self._parent),
            "destination": name_and_id_dict(self._child),
            "observer": name_and_id_dict(self.observer)
        })

        return d

    @staticmethod
    def create_and_append(parent: Processor, child: Processor, observer: Observer, tags=None, attributes=None):
        o = ProcessorsRelationPartOfObservation(parent, child, observer, tags, attributes)
        if parent:
            parent.observations_append(o)
        if child:
            child.observations_append(o)
        if observer:
            observer.observables_append(parent)
            observer.observables_append(child)
        return o

    @property
    def parent_processor(self):
        return self._parent

    @property
    def child_processor(self):
        return self._child

    @property
    def observer(self):
        return self._observer

    @staticmethod
    def partial_key(parent: Processor=None, child: Processor=None, observer: Observer=None):
        d = {"_t": RelationClassType.pp_isa.name}
        if child:
            d["__c"] = child.ident

        if parent:
            d["__p"] = parent.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self):
        d = {"_t": RelationClassType.pp_isa.name, "__p": self._parent.ident, "__c": self._child.ident}
        if self._observer:
            d["__oer"] = self._observer.ident
        return d


class ProcessorsRelationPartOfObservation(ProcessorsRelationObservation, Encodable):
    def __init__(self, parent: Processor, child: Processor, observer: Observer=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._parent = parent
        self._child = child
        self._observer = observer

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            "origin": name_and_id_dict(self._parent),
            "destination": name_and_id_dict(self._child),
            "observer": name_and_id_dict(self.observer)
        })

        return d

    @staticmethod
    def create_and_append(parent: Processor, child: Processor, observer: Optional[Observer] = None, tags=None, attributes=None):
        o = ProcessorsRelationPartOfObservation(parent, child, observer, tags, attributes)
        if parent:
            parent.observations_append(o)
        if child:
            child.observations_append(o)
        if observer:
            observer.observables_append(parent)
            observer.observables_append(child)
        return o

    @property
    def parent_processor(self):
        return self._parent

    @property
    def child_processor(self):
        return self._child

    @property
    def observer(self):
        return self._observer

    @staticmethod
    def partial_key(parent: Processor=None, child: Processor=None, observer: Observer=None):
        d = {"_t": RelationClassType.pp_part_of.name}
        if child:
            d["__c"] = child.ident

        if parent:
            d["__p"] = parent.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self):
        d = {"_t": RelationClassType.pp_part_of.name, "__p": self._parent.ident, "__c": self._child.ident}
        if self._observer:
            d["__oer"] = self._observer.ident
        return d


class ProcessorsRelationUndirectedFlowObservation(ProcessorsRelationObservation):
    """
    Represents an undirected Flow, from a source to a target Processor
    Undirected flow is DYNAMICALLY converted to Directed flow, for each factor:
    * If the factor of the source (parent) is "Incoming", the flow is from the parent to the child
    * If the factor of the source (parent) is "Outgoing", the flow is from the child to the parent

    from nexinfosys.models.musiasem_concepts import Processor, Observer, ProcessorsRelationUndirectedFlowObservation
    pr = ProcessorsRelationFlowObservation.create_and_append(Processor("A"), Processor("B"), Observer("S"))

    """
    def __init__(self, source: Processor, target: Processor, observer: Observer=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._source = source
        self._target = target
        self._observer = observer

    @staticmethod
    def create_and_append(source: Processor, target: Processor, observer: Observer, tags=None, attributes=None):
        o = ProcessorsRelationUndirectedFlowObservation(source, target, observer, tags, attributes)
        if source:
            source.observations_append(o)
        if target:
            target.observations_append(o)
        if observer:
            observer.observables_append(source)
            observer.observables_append(target)
        return o

    @property
    def source_processor(self):
        return self._source

    @property
    def target_processor(self):
        return self._target

    @property
    def observer(self):
        return self._observer

    @staticmethod
    def partial_key(source: Processor=None, target: Processor=None, observer: Observer=None):
        d = {"_t": RelationClassType.pp_undirected_flow.name}
        if target:
            d["__t"] = target.ident

        if source:
            d["__s"] = source.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self):
        d = {"_t": RelationClassType.pp_undirected_flow.name, "__s": self._source.ident, "__t": self._target.parent}
        if self._observer:
            d["__oer"] = self._observer.ident
        return d


class ProcessorsRelationUpscaleObservation(ProcessorsRelationObservation):
    def __init__(self, parent: Processor, child: Processor, observer: Observer, factor_name: str, quantity: str=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._parent = parent
        self._child = child
        self._observer = observer
        self._factor_name = factor_name
        self._quantity = quantity

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            "origin": name_and_id_dict(self._parent),
            "destination": name_and_id_dict(self._child),
            "observer": name_and_id_dict(self.observer),
            "interface": self.factor_name,
            "quantity": self.quantity
        })

        return d

    @staticmethod
    def create_and_append(parent: Processor, child: Processor, observer: Optional[Observer], factor_name: str, quantity: str, tags=None, attributes=None):
        o = ProcessorsRelationUpscaleObservation(parent, child, observer, factor_name, quantity, tags, attributes)
        if parent:
            parent.observations_append(o)
        if child:
            child.observations_append(o)
        if observer:
            observer.observables_append(parent)
            observer.observables_append(child)
        return o

    @property
    def parent_processor(self):
        return self._parent

    @property
    def child_processor(self):
        return self._child

    @property
    def factor_name(self):
        return self._factor_name

    @property
    def quantity(self):
        return self._quantity

    @property
    def observer(self):
        return self._observer

    @staticmethod
    def partial_key(parent: Processor=None, child: Processor=None, observer: Observer=None):
        d = {"_t": RelationClassType.pp_upscale.name}
        if child:
            d["__c"] = child.ident

        if parent:
            d["__p"] = parent.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self):
        d = {"_t": RelationClassType.pp_upscale.name, "__p": self._parent.ident, "__c": self._child.ident}
        if self._observer:
            d["__oer"] = self._observer.ident
        return d


class FactorsRelationDirectedFlowObservation(FactorsRelationObservation, Encodable):
    """
    Represents a directed Flow, from a source to a target Factor
    from nexinfosys.models.musiasem_concepts import Processor, Factor, Observer, FactorsRelationDirectedFlowObservation
    pr = ProcessorsRelationFlowObservation.create_and_append(Processor("A"), Processor("B"), Observer("S"))

    """

    def __init__(self, source: Factor, target: Factor, observer: Observer = None, weight: str=None, tags=None,
                 attributes=None, back: Factor = None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._source = source
        self._target = target
        self._back = back
        self._weight = weight
        self._observer = observer

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            "origin": name_and_id_dict(self.source_factor),
            "destination": name_and_id_dict(self.target_factor),
            "observer": name_and_id_dict(self.observer),
            "weight": self.weight
        })

        if self.back_factor:
            d.update({
                "back": name_and_id_dict(self.back_factor)
            })

        return d

    @staticmethod
    def create_and_append(source: Factor, target: Factor, observer: Optional[Observer], weight: str=None, tags=None,
                          attributes=None, back: Factor = None):
        o = FactorsRelationDirectedFlowObservation(source, target, observer, weight, tags, attributes, back=back)
        if source:
            source.observations_append(o)
        if target:
            target.observations_append(o)
        if back:
            back.observations_append(o)
        if observer:
            observer.observables_append(source)
            observer.observables_append(target)
            observer.observables_append(back)
        return o

    @property
    def source_factor(self):
        return self._source

    @property
    def target_factor(self):
        return self._target

    @property
    def back_factor(self):
        return self._back

    @property
    def weight(self):
        return self._weight

    @property
    def scale_change_weight(self):
        return self.attributes.get("scale_change_weight")

    @property
    def observer(self):
        return self._observer

    @staticmethod
    def partial_key(source: Factor = None, target: Factor = None, observer: Observer = None, back: Factor = None):
        d = {"_t": RelationClassType.ff_directed_flow.name}
        if target:
            d["__t"] = target.ident

        if source:
            d["__s"] = source.ident

        if back:
            d["__b"] = back.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self):
        d = {"_t": RelationClassType.ff_directed_flow.name,
             "__s": self._source.ident,
             "__t": self._target.ident,
             "__b": self._back.ident if self._back else None}
        if self._observer:
            d["__oer"] = self._observer.ident
        return d


class FactorsRelationScaleObservation(FactorsRelationObservation):
    def __init__(self, origin: Factor, destination: Factor, observer: Observer, quantity: str=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._origin = origin
        self._destination = destination
        self._observer = observer
        self._quantity = quantity

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            "origin": name_and_id_dict(self._origin),
            "destination": name_and_id_dict(self._destination),
            "observer": name_and_id_dict(self._observer),
            "quantity": self._quantity
        })

        return d

    @staticmethod
    def create_and_append(origin: Factor, destination: Factor, observer: Observer, quantity: str, tags=None, attributes=None):
        o = FactorsRelationScaleObservation(origin, destination, observer, quantity, tags, attributes)
        if origin:
            origin.observations_append(o)
        if destination:
            destination.observations_append(o)
        if observer:
            observer.observables_append(origin)
            observer.observables_append(destination)
        return o

    @property
    def origin(self):
        return self._origin

    @property
    def destination(self):
        return self._destination

    @property
    def quantity(self):
        return self._quantity

    @property
    def observer(self):
        return self._observer

    @staticmethod
    def partial_key(origin: Factor=None, destination: Factor=None, observer: Observer=None, origin_processor: Processor=None, destination_processor: Processor=None):
        d = {"_t": RelationClassType.ff_scale.name}
        if origin:
            d["__o"] = origin.ident
            d["__op"] = origin.processor.ident

        if destination:
            d["__d"] = destination.ident
            d["__dp"] = destination.processor.ident

        if origin_processor:
            d["__op"] = origin_processor.ident

        if destination_processor:
            d["__dp"] = destination_processor.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self):
        d = {"_t": RelationClassType.ff_scale.name,
             "__o": self._origin.ident, "__d": self._destination.ident,
             "__op": self._origin.processor.ident, "__dp": self._destination.processor.ident}
        if self._observer:
            d["__oer"] = self._observer.ident
        return d


# #################################################################################################################### #
# NUSAP PedigreeMatrix
# #################################################################################################################### #


class PedigreeMatrix(Nameable, Identifiable):
    """ A Pedigree Matrix, made of a list of lists
        Each list and each element of a list may have a description
    """
    PedigreeMatrixPhase = collections.namedtuple("PedigreeMatrixPhase", "code description modes")

    def __init__(self, name):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        self._codes = None  # Dict code to order
        self._phases = None

    def set_phases(self, codes, phases):
        """
        Initialize the pedigree matrix

        :param codes: list of codes
        :param phases: list of phases as parsed from the command
        :return:
        """
        self._codes = {int(c["mode"]): i for i, c in enumerate(codes)}
        self._phases = []
        for phase in phases:
            lst = []
            for i, mode in enumerate(phase):
                if i == 0:
                    continue
                lst.append((mode["mode"], ""))
            asp = PedigreeMatrix.PedigreeMatrixPhase(code=phase[0]["mode"], description=phase[0]["description"], modes=lst)
            self._phases.append(asp)

    def get_modes_for_code(self, code: str):
        """
        From a string of integers return a list of modes

        :param code:
        :return:
        """
        lst = []
        code = str(code)
        for i in range(len(code)):
            asp = self._phases[i]
            idx = self._codes[int(code[i])]
            lst.append(asp.modes[idx][0])
        return lst

    def get_phase(self, aspect: int):
        """
        From the aspect number (starting at 1) get the code

        :param aspect:
        :return:
        """
        asp = self._phases[aspect - 1]
        return asp.code

    def get_phase_mode(self, phase: int, mode: int):
        """
        From the phase number and the mode inside the phase, get the code

        """
        asp = self._phases[phase - 1]
        ret = asp.modes[self._codes[mode]][0]
        return ret

    @staticmethod
    def partial_key(name: str=None, ident: str=None):
        d = dict(_t="pm")
        if name:
            d["_n"] = name
        if ident:
            d["__id"] = ident
        return d

    def key(self):
        return {"_t": "pm", "_n": self.name, "__id": self.ident}


# #################################################################################################################### #
# PARAMETERS, SOLVING, BENCHMARKS, INDICATORS
# #################################################################################################################### #


class Parameter(Nameable, Identifiable, Encodable):
    """ A numeric variable changeable directly by an analyst """
    def __init__(self, name):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        self._type = None
        self._range = None
        self._description = None
        self._default_value = None
        self._group = None
        self._current_value = None

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            'type': self._type,
            'range': self._range,
            'description': self._description,
            'default_value': self._default_value,
            'group': self._group
        })

        return d

    @property
    def type(self):
        return self._type

    @property
    def default_value(self):
        return self._default_value

    @property
    def current_value(self):
        return self._current_value

    @current_value.setter
    def current_value(self, value):
        self._current_value = value

    @property
    def group(self):
        return self._group

    @staticmethod
    def partial_key(name: str=None):
        d = dict(_t="param")
        if name:
            d["_n"] = name
        return d

    def key(self):
        return {"_t": "param", "_n": self.name, "__o": self.ident}


class ProblemStatement(Identifiable, Encodable):
    """
    Contains the parameters for the different scenarios, plus parameters needed to launch a solving process
    """
    def __init__(self, solving_parameters=None, scenarios=None):
        Identifiable.__init__(self)
        # Parameters characterizing the solver to be used and the parameters it receives
        self._solving_parameters = ifnull(solving_parameters, {})  # type: Dict[str, str]
        # Each scenario is a set of parameter values (expressions)
        self._scenarios = ifnull(scenarios, {})  # type: Dict[str, Dict[str, str]]

    def encode(self):
        d = Encodable.parents_encode(self, __class__)
        # d.update({})
        return d

    @staticmethod
    def partial_key():
        d = dict(_t="ps")
        return d

    def key(self):
        """
        Return a Key for the identification of the ProblemStatement in the registry

        :param registry:
        :return:
        """
        return {"_t": "ps", "__id": self.ident}

    @property
    def solving_parameters(self):
        return self._solving_parameters

    @property
    def scenarios(self):
        return self._scenarios


class BenchmarkGroup(Enum):
    feasibility = 1
    viability = 2
    desirability = 3


class Benchmark(Nameable, Identifiable, Encodable):
    """ Used to frame numbers, either to make comparisons or to check a constraint """
    def __init__(self, name, benchmark_group, stakeholders: List[str]):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        self._benchmark_group = benchmark_group
        self._stakeholders = stakeholders
        self._ranges = create_dictionary()

    def encode(self):
        d = {
            'benchmark_group': self._benchmark_group,
            'ranges': self._ranges
        }
        return d

    @property
    def benchmark_group(self):
        return self._benchmark_group

    @property
    def stakeholders(self):
        return self._stakeholders

    @property
    def ranges(self):
        # TODO Search in which category falls the numeric value "v", and return the corresponding tuple
        return self._ranges

    @staticmethod
    def partial_key(name: str=None):
        d = dict(_t="b")
        if name:
            d["_n"] = name
        return d

    def key(self):
        """
        Return a Key for the identification of the Benchmark in the registry

        :param registry:
        :return:
        """
        return {"_t": "b", "_n": self.name, "__id": self.ident}


class IndicatorCategories(Enum):  # Used in FlowFund
    factor_types_expression = 1  # It can be instantiated into several processors, generating "factors_expression" indicators. At the same time, it may serve to compute the accumulated of these processors
    factors_expression = 2  # An example is a Metabolic Rate of a processor. Can be originated from a "factor_types_expression"
    case_study = 3  # An expression operating on factors of from different parts of the case study


class Indicator(Nameable, Identifiable, Encodable):
    """ An arithmetic expression resulting in a numeric value to assess a quality of the SES (SocioEcological System) under study
    Categorize indicators:
    * Attached to FactorType
    * Attached to Processor
    * Attached to CaseStudy
    """
    def __init__(self, name: str, formula: str, from_indicator: Optional["Indicator"], processors_selector: str,
                 benchmarks: List[Benchmark], indicator_category: IndicatorCategories, description=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        self._formula = formula
        self._from_indicator = from_indicator
        self._processors_selector = processors_selector
        self._benchmarks = benchmarks
        self._indicator_category = indicator_category
        self._description = description

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            'formula': self._formula,
            'from_indicator': self._from_indicator,
            'processors_selector': self._processors_selector,
            'benchmarks': self._benchmarks,
            'indicator_category': getattr(self._indicator_category, "name", None),
            'description': self._description
        })

        return d

    @property
    def processors_selector(self):
        return self._processors_selector

    @property
    def formula(self):
        return self._formula

    @staticmethod
    def partial_key(name: str=None):
        d = dict(_t="i")
        if name:
            d["_n"] = name

        return d

    def key(self):
        """
        Return a Key for the identification of the Hierarchy in the registry

        :param registry:
        :return:
        """
        return {"_t": "i", "_n": self.name, "__id": self.ident}


class MatrixIndicator(Nameable, Identifiable, Encodable):
    """ An arrangement of the results of accounting, where rows are a subset of all processors, and columns
        can be InterfaceTypes, Indicators (scalar), and Attributes (of the processor)
    Categorize indicators:
    * Attached to FactorType
    * Attached to Processor
    * Attached to CaseStudy
    """
    # TODO Expressions should support rich selectors, of the form "factors from processors matching these properties
    #  and factor types with those properties (include "all factors in a processor") AND/OR ..." plus set operations
    #  (union/intersection).
    # TODO Then, compute an operation over all selected factors.
    # TODO To generate multiple instances of the indicator or a single indicator accumulating many things.
    def __init__(self, name: str, context_type: str, processors_selector: str, interfaces_selector: str,
                 indicators_selector: str, attributes_selector: str, description=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        self._context_type = context_type
        self._processors_selector = processors_selector
        self._interfaces_selector = interfaces_selector
        self._indicators_selector = indicators_selector
        self._attributes_selector = attributes_selector
        self._description = description

    def encode(self):
        d = Encodable.parents_encode(self, __class__)

        d.update({
            'context_type': self._context_type,
            'processors_selector': self._processors_selector,
            'interfaces_selector': self._interfaces_selector,
            'indicators_selector': self._indicators_selector,
            'attributes_selector': self._attributes_selector,
            'description': self._description
        })

        return d

    @property
    def scope(self):
        return self._context_type

    @property
    def processors_selector(self):
        return self._processors_selector

    @property
    def interfaces_selector(self):
        return self._interfaces_selector

    @property
    def description(self):
        return self._description

    @staticmethod
    def partial_key(name: str=None):
        d = dict(_t="mi")
        if name:
            d["_n"] = name

        return d

    def key(self):
        """
        Return a Key for the identification of the Hierarchy in the registry

        :param registry:
        :return:
        """
        return {"_t": "mi", "_n": self.name, "__id": self.ident}


# #################################################################################################################### #
# MAPPING, EXTERNAL DATASETS
# #################################################################################################################### #


class Mapping:
    """
    Transcription of information specified by a mapping command
    """
    def __init__(self, name, source, dataset, origin, destination, the_map: List[Tuple]):
        self.name = origin + " -> " + destination
        self.name = name
        self.source = source  # Which Observer produced the mapping
        self.dataset = dataset  # Optionally, a dataset containing the ORIGIN hierarchy
        self.origin = origin  # Hierarchy (Dataset Dimensions are Hierarchies)
        self.destination = destination  # Destination Hierarchy (Dataset Dimensions are Hierarchies)
        # the_map is of the form:
        # [{"o": "", "to": [{"d": "", "w": "", "ctx": <id>}]}]
        # [ {o: origin category, to: [{d: destination category, w: weight assigned to destination category}] } ]
        # It is used by the ETL load dataset command
        self.map = the_map  # List of tuples (pairs) source code, destination code[, expression]

# TODO Currently ExternalDataset is used only in evaluation of expressions. It may be removed because
# TODO expressions can refer directly to State -> datasets


class ExternalDataset:
    def __init__(self, name, ds: pd.DataFrame):
        self._name = name
        self._ds = None  # Dataframe containing the Dataset
        self._persistent_dataset = None  # Reference to persistent Dataset, in case
        self._function_returning_dataset = None  # A dataset can be delegated to a function

    def get_dimensions(self):
        return self._ds.index.columns

    def get_columns(self):
        return self._ds.columns

    def get_data(self, select, filter_, aggregate, order, func_params=None):
        """
        Function responsible of obtaining part of previously gathered dataset, which is stored in memory in a Dataframe

        Because a dataset can be dynamically obtained by a function, it has the optional "func_params" parameter,
        containing a list of parameters which can be both args and kwargs (for the second, tuples of two elements are expected)

        :param select: Similar to SQL's SELECT
        :param filter_: Similar to SQL's WHERE
        :param aggregate: Similar to SQL's GROUP BY
        :param order: Similar to SQL's ORDER BY
        :param params:
        :return: The response
        """
        return None  # ExternalDataset(None, ds)

# ######################################################################################################################
# IN-MEMORY STATISTICAL DATASET
# ######################################################################################################################


class MConcept(Nameable, Qualifiable):
    def __init__(self, name, attribute_type=None, dataset=None, attributes=None):
        Nameable.__init__(self, name)
        Qualifiable.__init__(self, attributes)
        # Contains the type of the concept. Also the domain of the concept which in the case of a Dimension will be a Hierarchy
        self._attribute_type = None  # type: AttributeType
        self._dataset = None  # type: MDataset


class MDimensionConcept(MConcept):
    def __init__(self):
        self._is_time = None


class MMeasureConcept(MConcept):
    def __init__(self, name, attributes):
        MConcept.__init__(self, name, attributes)


class MAttributeConcept(MConcept):
    def __init__(self, name, attributes):
        MConcept.__init__(self, name, attributes)


class MDataSource(Nameable, Identifiable, Qualifiable):
    def __init__(self, name, description=None, attributes=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        Qualifiable.__init__(self, attributes)
        self._description = description
        self._metadata = None  # Here is the place for metadata not in other fields. Not used for now.
        self._databases = []  # type: List[MDatabase]


class MDatabase(Nameable, Identifiable, Qualifiable):
    def __init__(self, name, description=None, attributes=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        Qualifiable.__init__(self, attributes)
        self._description = description
        self._metadata = None  # Here is the place for metadata not in other fields. Not used for now.
        self._data_source = None  # type: MDataSource
        self._datasets = []  # type: List[MDataset]


class MDataset(Nameable, Identifiable, Qualifiable):
    def __init__(self, name, description=None, attributes=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        Qualifiable.__init__(self, attributes)
        self._description = description
        self._concepts_list = []  # type: List[MConcept]
        self._database = None  # type: MDatabase
        self._metadata = None  # Here is the place for metadata not in other fields. **Not used for now**
        self._data = None  # type: pd.DataFrame
