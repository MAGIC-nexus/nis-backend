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
import urllib
from collections import OrderedDict
from enum import Enum
from typing import *  # Type hints
from uuid import UUID
import pint  # Physical Units management
import pandas as pd
import logging
from attr import attrs, attrib

from backend.common.helper import create_dictionary, strcmp, PartialRetrievalDictionary, \
    case_sensitive, is_boolean, is_integer, is_float, is_datetime, is_url, is_uuid, to_datetime, to_integer, to_float, \
    to_url, to_uuid, to_boolean, to_category, to_str, is_category, is_str, is_geo, to_geo, ascii2hex, \
    Encodable
from backend.models import ureg, log_level
from backend.model_services import State, get_case_study_registry_objects, LocallyUniqueIDManager
from backend.models import CodeImmutable

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
            "ident": ascii2hex(self.ident)
        }

    @property
    def ident(self):
        return self._id


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


class Taggable(Encodable):
    """ A concept with a set of tags """
    def __init__(self, tags):
        self._tags = set()
        if tags:
            self.tags_append(tags)

    def encode(self):
        return {
            "tags": self.tags
        }

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
    """ An entity with a dictionary of Attributes """
    def __init__(self, attributes=None):
        # "name" property of AttributeType -> to Value
        self._attributes = create_dictionary()  # type: Dict[str, object]
        # "name" property of AttributeType -> to AttributeType
        self._name_to_attribute_type = create_dictionary()  # type: Dict[str, AttributeType]
        if attributes:
            for k in attributes:
                self.attributes_append(k, attributes[k])
                # TODO From the attribute name obtain the AttributeType (in the global registry)

    def encode(self):
        return {
            "attributes": self.attributes
        }

    @property
    def attributes(self):
        return self._attributes

    def attributes_append(self, name, value, attribute_type=None):
        if value:
            self._attributes[name] = value
        if attribute_type:
            self._name_to_attribute_type[name] = attribute_type


class Observable(Encodable):
    """ An entity which can be structurally (relations with other observables) or quantitatively observed.
        It can have none to infinite possible observations
    """
    def __init__(self, location: "Geolocation"):
        self._location = location  # Definition of where the observable is
        self._physical_nature = None
        self._observations = []  # type: List[Union[FactorQuantitativeObservation, RelationObservation]]

    def encode(self):
        return {
            "observations": self.observations
        }

    # Methods to manage the properties
    @property
    def observations(self):
        return self._observations

    def observations_append(self, observation):
        if isinstance(observation, (list, set)):
            lst = observation
        else:
            lst = [observation]

        self._observations.extend(lst)


class HierarchyNode(Nameable, Encodable):
    """ Taxon, Processor and Factor
        A hierarchy node can be a member of several hierarchies ¿what's the name of this property?
        A hierarchy can be flat, so a hierarchy node can be member of a simple list
    """

    def __init__(self, name, parent=None, parent_weight=1.0, hierarchy=None, level=None, label=None, description=None, referred_node=None):
        Nameable.__init__(self, name)
        self._parents = []
        self._parents_weights = []
        self._referred_node = referred_node  # type: HierarchyNode in another Hierarchy
        self._children = set()  # type: HierarchyNode in the same "hierarchy"
        self._level = level  # type: HierarchyLevel
        self._hierarchy = hierarchy  # type: Hierarchy
        if referred_node and referred_node.hierarchy == self.hierarchy:
            raise Exception("The hierarchy of a node and the hierarchy of the referred cannot be the same")
        if hierarchy:  # Add name to the hierarchy
            hierarchy.codes[name] = self
        if parent:
            self.set_parent(parent, parent_weight)

        self._label = label
        self._description = description

    def encode(self):
        encoded_referred_node = None
        if self.referred_node:
            encoded_referred_node = {
                "name": self.referred_node.name,
                "hierarchy": self.referred_node.hierarchy.name
            }

        encoded_parent_node = None
        if self.parent:
            encoded_parent_node = self.parent.name

        d = {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "parent": encoded_parent_node,
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
    def __init__(self, name, region_name=None, projection=None, shape=None):
        self._name = name
        self._region_name = region_name
        self._projection = projection
        self._shape = shape

    def __eq__(self, other):
        return self.region_name == other.region_name and \
               self.projection == other.projection and \
               self.shape == other.shape

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def region_name(self):
        return self._region_name

    @region_name.setter
    def region_name(self, region_name):
        self._region_name = region_name

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, projection):
        self._projection = projection

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape


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


class QualifiedQuantityExpression:
    """ The base class for quantitative observations
    """
    def __init__(self, e: Union[str, dict]):
        self._expression = e

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, expression):
        self._expression = expression

    @staticmethod
    def n(n: float):
        """ "N" of NUSAP """
        return QualifiedQuantityExpression(json.dumps({'n': n}))

    @staticmethod
    def nu(n: float, u: str):
        """ "NU" of NUSAP """
        # Check that "u" is a recognized unit type
        try:
            ureg(u)
        except pint.errors.UndefinedUnitError:
            # The user should know that the specified unit is not recognized
            raise Exception("The specified unit '" + u + "' is not recognized")
        return QualifiedQuantityExpression(json.dumps({'n': n, 'u': u}))

    def __repr__(self):
        return str(self.expression)

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

        d = {
            "description": self._description,
            "hierarchy_group": None if not self.hierarchy_group else self.hierarchy_group.encode(),
            "nodes": hierarchy_to_list(self.roots)
        }

        d.update(Encodable.parents_encode(self, __class__))

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
    def __init__(self, expression: QualifiedQuantityExpression=None):
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
    def expression(self, e: QualifiedQuantityExpression=None):
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
        d = {
            "description": self.description
        }

        d.update(Encodable.parents_encode(self, __class__))
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

    Parameters, ScaleChangers, Instantiators, ETL or dataset transforms -and other possible adaptive
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


class ReferenceType(Enum):
    provenance = 2,
    geographic = 3,
    bibliography = 4,


class Reference(Nameable, Identifiable, Qualifiable):
    """ A dictionary containing a set of key-value pairs
        with some validation schema (search code for "ref_prof" global variable)
    """
    def __init__(self, name, ref_type, content):
        """

        :param name:
        :param ref_type: One of the elements declared in "ref_prof"
        :param content:
        """
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        Qualifiable.__init__(self, content)
        self._ref_type = ref_type

    # TODO A method to validate that attributes

    @staticmethod
    def partial_key(name: str=None, ref_type: str=None):
        d = dict(_t="r")
        if name:
            d["_n"] = name
        if ref_type:
            d["_rt"] = ref_type
        return d

    def key(self):
        return {"_t": "r", "_n": self.name, "_rt": self._ref_type, "__o": self.ident}


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


class Observer(Identifiable, Nameable):
    """ An entity capable of producing Observations on an Observable
        In our context, it is a process obtaining data
    """
    no_observer_specified = "_default_observer"

    def __init__(self, name, description=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        self._description = description  # Informal description of the observation process (it could be just a manual transcription of numbers from a book)
        self._observation_process_description = None  # Formal description of the observation process
        self._sources = []  # type: List[Source]
        self._observables = []  # type: List[Observable]

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

    @property
    def sources(self):
        return self._sources

    def sources_append(self, source):
        if isinstance(source, (list, set)):
            lst = source
        else:
            lst = [source]

        self._sources.extend(lst)

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


class FactorType(Identifiable, HierarchyNode, HierarchyExpression, Taggable, Qualifiable, Encodable):  # Flow or fund type (not attached to a Processor)
    """ A Factor as type, in a hierarchy, a Taxonomy """
    def __init__(self, name, parent=None, hierarchy=None,
                 tipe: FlowFundRoegenType=FlowFundRoegenType.flow,
                 tags=None, attributes=None, expression=None,
                 orientation=None, opposite_processor_type=None):
        Identifiable.__init__(self)
        HierarchyNode.__init__(self, name, parent, hierarchy=hierarchy)
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        HierarchyExpression.__init__(self, expression)
        self._roegen_type = tipe
        self._orientation = orientation
        self._opposite_processor_type = opposite_processor_type
        self._physical_type = None  # TODO Which physical types. An object
        self._default_unit_str = None  # TODO A string representing the unit, compatible with the physical type
        self._factors = []

    def encode(self):
        d = {
            'roegen_type': self.roegen_type.name,
            'orientation': self.orientation
        }

        d.update(Encodable.parents_encode(self, __class__))

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
    def orientation(self):
        return self._orientation

    @property
    def opposite_processor_type(self):
        return self._opposite_processor_type

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


class Processor(Identifiable, Nameable, Taggable, Qualifiable, Automatable, Observable, Encodable):
    def __init__(self, name, external: bool=False,
                 location: "Geolocation"=None, tags=None, attributes=None,
                 referenced_processor: "Processor"=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        Observable.__init__(self, location)

        self._factors = []  # type: List[Factor]
        self._relationships = []  # type: List[ProcessorsRelationObservation]
        self._local_indicators = []  # type: List[Indicator]

        self._type = None  # Environment, Society
        self._external = external  # Either external (True) or internal (False)
        self._stock = None  # True, False
        self._instantiation_type = None  # Instance, "Unit processor" (Intensive Processor), "Function"

        # The processor references a previously defined processor
        # * If a factor is defined, do not look in the referenced
        # * If a factor is not defined here, look in the referenced
        # * Other properties can also be referenced (if defined locally it is not searched in the referenced)
        # * FactorObservations are also assimilated
        #   (although expressions evaluated in the context of the local processor, not the referenced one)
        # * Relations are NOT assimilated, at least by default. THIS HAS TO BE STUDIED IN DEPTH
        self._referenced_processor = referenced_processor

    def encode(self):
        d = {
            'type': self.type_processor,
            'external': self.external,
            'stock': self.stock,
            'instantiation_type': self.instantiation_type,
            'interfaces': self.factors
        }

        d.update(Identifiable.encode(self))
        d.update(Nameable.encode(self))
        d.update(Taggable.encode(self))
        d.update(Qualifiable.encode(self))
        d.update(Automatable.encode(self))

        #d.update(Encodable.parents_encode(self, __class__))

        return d

    @property
    def instantiation_type(self):
        return self._instantiation_type

    @property
    def stock(self):
        return self._stock

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

    @property
    def extensive(self):
        # TODO True if of all values defined for all factors no value depends on another factor
        return False

    @property
    def intensive(self):
        return not self.extensive

    @property
    def external(self):
        tmp = self._external
        if tmp is None and self.referenced_processor:
            tmp = self.referenced_processor.external

        return tmp

    @property
    def internal(self):
        return not self._external

    @property
    def type_processor(self):
        # True if the processor is a type. The alternative is the processor being real. Type abstracts a function
        # "type" and function are similar. But it depends on the analyst, who can also define the processor as
        # structural and "type" at the same time
        tmp = self._type
        if tmp is None and self.referenced_processor:
            tmp = self.referenced_processor.type_processor

        return tmp

    @type_processor.setter
    def type_processor(self, v: bool):
        self._type = v

    @property
    def real_processor(self):
        return not self.type_processor

    @real_processor.setter
    def real_processor(self, v: bool):
        self.type_processor = not v

    def simple_name(self):
        parts = self.name.split(".")
        if len(parts) > 1:
            return parts[-1]
        else:
            return self.name

    def full_hierarchy_names(self, registry: PartialRetrievalDictionary):
        """
        Obtain the full hierarchy name of the current processor
        It looks for the PART-OF relations in which the processor is in the child side

        :param registry:
        :return:
        """
        # Get matching relations
        part_of_relations = registry.get(ProcessorsRelationPartOfObservation.partial_key(child=self))

        # Compose the name, recursively
        if len(part_of_relations) == 0:
            return [self.name]
        else:
            # Take last part of the name
            last_part = self.name.split(".")[-1]
            return [(p+"."+last_part) for rel in part_of_relations for p in rel.parent_processor.full_hierarchy_names(registry)]

    def clone(self, state: Union[PartialRetrievalDictionary, State], objects_processed: dict=None, level=0):
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
        :param objects_processed: Dictionary containing already processed Processors and Factors (INTERNAL USE)
        :param level: Recursion level (INTERNAL USE)
        :return:
        """
        if isinstance(state, PartialRetrievalDictionary):
            glb_idx = state
        else:
            glb_idx, _, _, _, _ = get_case_study_registry_objects(state)

        # Create new Processor
        p = Processor(self.name,
                      external=self.external,
                      location=self._location,
                      tags=self._tags,
                      attributes=self._attributes,
                      referenced_processor=self.referenced_processor
                      )
        p._type = self._type
        p._external = self._external
        p._stock = self._stock

        glb_idx.put(p.key(), p)

        if not objects_processed:
            objects_processed = {}

        objects_processed[self] = p

        # Factors
        for f in self.factors:
            f_ = Factor.clone_and_append(f, p) # The Factor is cloned and appended into the Processor "p"
            glb_idx.put(f_.key(), f_)
            if f not in objects_processed:
                objects_processed[f] = f_
            else:
                raise Exception("Unexpected: the factor "+f.processor.name+":"+f.taxon.name+" should not be cloned again.")

        # Local indicators
        for li in self._local_indicators:
            formula = li._formula  # TODO Adapt formula of the new indicator to the new Factors
            li_ = Indicator(li._name, formula, li, li._benchmark, li._indicator_category)
            self._local_indicators.append(li_)

        # Clone child Processors (look for part-of relations)
        part_of_relations = glb_idx.get(ProcessorsRelationPartOfObservation.partial_key(parent=self))
        for rel in part_of_relations:
            if rel.child_processor not in objects_processed:
                p2 = rel.child_processor.clone(state, objects_processed, level+1)  # Recursive call
                objects_processed[rel.child_processor] = p2
            else:
                p2 = objects_processed[rel.child_processor]

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
            for o in objects_processed:
                if isinstance(o, Factor):
                    for f in glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key(source=o)): # As Source
                        if f not in considered_flows:
                            if f.target_factor in objects_processed:
                                new_f = FactorsRelationDirectedFlowObservation(source=o, target=objects_processed[f.target_factor], observer=f.observer, weight=f.weight, tags=f.tags, attributes=f.attributes)
                            else:
                                new_f = FactorsRelationDirectedFlowObservation(source=o, target=f.target_factor, observer=f.observer, weight=f.weight, tags=f.tags, attributes=f.attributes)
                            glb_idx.put(new_f.key(), new_f)
                            considered_flows.add(f)
                    for f in glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key(target=o)): # As Target
                        if f not in considered_flows:
                            if f.source_factor in objects_processed:
                                new_f = FactorsRelationDirectedFlowObservation(source=objects_processed[f.source_factor], target=o, observer=f.observer, weight=f.weight, tags=f.tags, attributes=f.attributes)
                            else:
                                new_f = FactorsRelationDirectedFlowObservation(source=f.source_factor, target=o, observer=f.observer, weight=f.weight, tags=f.tags, attributes=f.attributes)

                            glb_idx.put(new_f.key(), new_f)
                            considered_flows.add(f)
                else:  # "o" is a Processor
                    for f in glb_idx.get(ProcessorsRelationUndirectedFlowObservation.partial_key(source=o)): # As Source
                        if f not in considered_flows:
                            if f.target_factor in objects_processed:
                                new_f = ProcessorsRelationUndirectedFlowObservation(source=o, target=objects_processed[f.target_factor], observer=f.observer, weight=f.weight, tags=f.tags, attributes=f.attributes)
                            else:
                                new_f = ProcessorsRelationUndirectedFlowObservation(source=o, target=f.target_factor, observer=f.observer, weight=f.weight, tags=f.tags, attributes=f.attributes)
                            glb_idx.put(new_f.key(), new_f)
                            considered_flows.add(f)
                    for f in glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key(target=o)): # As Target
                        if f not in considered_flows:
                            if f.source_factor in objects_processed:
                                new_f = ProcessorsRelationUndirectedFlowObservation(source=objects_processed[f.source_factor], target=o, observer=f.observer, weight=f.weight, tags=f.tags, attributes=f.attributes)
                            else:
                                new_f = ProcessorsRelationUndirectedFlowObservation(source=f.source_factor, target=o, observer=f.observer, weight=f.weight, tags=f.tags, attributes=f.attributes)

                            glb_idx.put(new_f.key(), new_f)
                            considered_flows.add(f)

        return p

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


class Factor(Identifiable, Nameable, Taggable, Qualifiable, Observable, Automatable, Encodable):
    """ A Flow or Fund, when attached to a Processor
        It is automatable because an algorithm emulating an expert could inject Factors into Processors (as well as
        associated Observations)
    """
    def __init__(self, name, processor: Processor, in_processor_type: FactorInProcessorType, taxon: FactorType=None,
                 referenced_factor: "Factor" = None,
                 location: "Geolocation"=None, tags=None, attributes=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Observable.__init__(self, location)
        Automatable.__init__(self)
        self._processor = processor
        self._taxon = taxon
        self._type = in_processor_type

        # The factor references a previously defined factor
        # * It can contain its own observations
        # * Inherits all the None properties
        # * Relations are NOT assimilated, at least by default. THIS HAS TO BE STUDIED IN DEPTH
        self._referenced_factor = referenced_factor

    def encode(self):
        d = {
            'type': self.type,
            'interface_type': self.taxon
        }

        d.update(Encodable.parents_encode(self, __class__))

        return d

    @property
    def processor(self):
        return self._processor

    @property
    def referenced_factor(self):
        return self._referenced_factor

    @staticmethod
    def create_and_append(name, processor: Processor, in_processor_type: FactorInProcessorType, taxon: FactorType,
                          location: "Geolocation"=None, tags=None, attributes=None):
        f = Factor(name, processor, in_processor_type, taxon, location, tags, attributes)
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
    def taxon(self):  # Factor Type
        tmp = self._taxon
        if tmp is None and self.referenced_factor:
            tmp = self.referenced_factor.taxon

        return tmp

    @property
    def roegen_type(self):
        tmp = self._taxon.roegen_type
        if tmp is None and self.referenced_factor:
            tmp = self.referenced_factor.roegen_type

        return tmp

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
                                     location=factor._location, tags=factor._tags, attributes=factor._attributes)
        # Observations (only Quantities, because they are owned by the Factor)
        for o in factor.observations:
            if isinstance(o, FactorQuantitativeObservation):
                FactorQuantitativeObservation.create_and_append(o.value, o.factor, o.observer, o.tags, o.attributes)

        return f

    @staticmethod
    def partial_key(processor: Processor=None, factor_type: FactorType=None):
        d = {"_t": "f"}
        if processor:
            d["__p"] = processor.ident
        if factor_type:
            d["__ft"] = factor_type.ident

        return d

    def key(self):
        """
        Return a Key for the identification of the Factor in the registry
        :return:
        """
        return {"_t": "f", "__id": self.ident, "__p": self._processor.ident, "__ft": self._taxon.ident}


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
    pp_part_of = 1
    pp_undirected_flow = 2
    pp_upscale = 3
    pp_isa = 4
    pp_aggregate = 5
    pp_associate = 6

    ff_directed_flow = 10
    ff_reverse_directed_flow = 11
    ft_directed_linear_transform = 12


class FactorQuantitativeObservation(Taggable, Qualifiable, Automatable, Encodable):
    """ An expression or quantity assigned to an Observable (Factor) """
    def __init__(self, v: QualifiedQuantityExpression, factor: Factor=None, observer: Observer=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._value = v
        self._factor = factor
        self._observer = observer

    def encode(self):
        d = {
            "value": self.value,
            #"interface": self.factor,
            #"observer": self.observer
        }
        return d

    @staticmethod
    def create_and_append(v: QualifiedQuantityExpression, factor: Factor, observer: Observer, tags=None, attributes=None):
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

    @staticmethod
    def partial_key(factor: Factor=None, observer: Observer=None):
        d = {"_t": "qq"}
        if factor:
            d["__f"] = factor.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self):
        d = {"_t": "qq", "__f": self._factor.ident}
        if self._observer:
            d["__oer"] = self._observer.ident
        return d


class RelationObservation(Taggable, Qualifiable, Automatable):  # All relation kinds
    pass


class ProcessorsRelationObservation(RelationObservation):  # Just base of ProcessorRelations
    pass


class FactorTypesRelationObservation(RelationObservation):  # Just base of FactorTypesRelations
    pass


class FactorsRelationObservation(RelationObservation):  # Just base of FactorRelations
    pass


class FactorTypesLT_Specific:
    def __init__(self, alpha: Union[float, str], origin_unit: str=None, dest_unit: str=None,
                 src_proc_context_qry: str=None, dst_proc_context_qry: str=None,
                 observer: Observer=None):
        """
        A specific "alpha" to perform a linear change of scale between an origin FactorType and a destination FactorType
        It can contain the context where this "alpha" can be applied. The context -for now- is relative to the known FactorType:
        *
        :param alpha:
        :param origin_unit:
        :param dest_unit:
        :param src_proc_context_qry:
        :param dst_proc_context_qry:
        :param observer:
        """
        self.alpha = alpha
        self.origin_unit = origin_unit
        self.dest_unit = dest_unit
        self.src_proc_context_qry = src_proc_context_qry
        self.dst_proc_context_qry = dst_proc_context_qry
        self.observer = observer


class FactorTypesRelationUnidirectionalLinearTransformObservation(FactorTypesRelationObservation):
    """
    Expression of an Unidirectional Linear Transform, from an origin FactorType to a destination FactorType
    A weight can be a expression containing parameters
    This relation will be applied to Factors which are instances of the origin FactorTypes, to obtain destination
    FactorTypes
    """
    def __init__(self, origin: FactorType, destination: FactorType, generate_back_flow: bool=False, weight: Union[float, str]=None, origin_context=None, destination_context=None, observer: Observer=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        # TODO Back flow with proportional weight?? or weight "1.0"?
        self._generate_back_flow = generate_back_flow  # True: generate a FactorsRelationDirectedFlowObservation in the opposite direction, if not already existent
        self._scales = []  # type: List[FactorTypesLT_Specific]
        if weight:
            tmp = FactorTypesLT_Specific(weight, origin_unit=None, dest_unit=None, )
        self._origin = origin
        self._destination = destination
        self._weight = weight
        self._observer = observer
        self._origin_context = origin_context
        self._destination_context = destination_context

    @staticmethod
    def create_and_append(origin: FactorType, destination: FactorType, weight, origin_context=None, destination_context=None, observer: Observer=None, tags=None, attributes=None):
        o = FactorTypesRelationUnidirectionalLinearTransformObservation(origin, destination, weight, origin_context, destination_context, observer, tags, attributes)
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
    def observer(self):
        return self._observer

    @staticmethod
    def partial_key(origin: FactorType=None, destination: FactorType=None, observer: Observer=None):
        d = {"_t": RelationClassType.ft_directed_linear_transform.name}
        if origin:
            d["__o"] = origin.ident

        if destination:
            d["__d"] = destination.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self):
        d = {"_t": RelationClassType.ft_directed_linear_transform.name,
             "__o": self._origin.ident, "__d": self._destination.ident}
        if self._observer:
            d["__oer"] = self._observer.ident
        return d


class ProcessorsRelationIsAObservation(ProcessorsRelationObservation):
    def __init__(self, parent: Processor, child: Processor, observer: Observer=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._parent = parent
        self._child = child
        self._observer = observer

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


class ProcessorsRelationPartOfObservation(ProcessorsRelationObservation):
    def __init__(self, parent: Processor, child: Processor, observer: Observer=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._parent = parent
        self._child = child
        self._observer = observer

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

    from backend.models.musiasem_concepts import Processor, Observer, ProcessorsRelationUndirectedFlowObservation
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

    @staticmethod
    def create_and_append(parent: Processor, child: Processor, observer: Observer, factor_name: str, quantity: str, tags=None, attributes=None):
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

    def activate(self, state):
        """
        Materialize the relation into something computable.
        In this case, create an Factor (if it does not exist) and an Observation associated to it

        :param state:
        :return:
        """
        # TODO Look for the factor in the child processor
        processor = self._child
        factor = None
        for f in processor.factors:
            if strcmp(f.name, self._factor_name) and f.processor == processor:
                factor = f
                break
        # If the factor is not found, create it
        if not factor:
            factor = Factor(self._factor_name, processor)
            # TODO Store Factor in the global index glb_idx.put(f_key, f)

        # Add the observation. The value is an expression involving the factor of the parent
        parent_name = self._parent.full_hierarchy_names()
        if len(parent_name) > 0:
            parent_name = parent_name[0]
            if len(parent_name) > 1:
                logger
        fo = FactorQuantitativeObservation.create_and_append(v=QualifiedQuantityExpression(parent_name + ":" + self._factor_name+" * ("+self._quantity+")"),
                                                             factor=f,
                                                             observer=self._observer,
                                                             tags=None,
                                                             attributes={"relative_to": None,
                                                                         "time": None,
                                                                         "geolocation": None,
                                                                         "spread": None,
                                                                         "assessment": None,
                                                                         "pedigree": None,
                                                                         "pedigree_template": None,
                                                                         "comments": None
                                                                         }
                                                             )

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


class FactorsRelationDirectedFlowObservation(FactorsRelationObservation):
    """
    Represents a directed Flow, from a source to a target Factor
    from backend.models.musiasem_concepts import Processor, Factor, Observer, FactorsRelationDirectedFlowObservation
    pr = ProcessorsRelationFlowObservation.create_and_append(Processor("A"), Processor("B"), Observer("S"))

    """

    def __init__(self, source: Factor, target: Factor, observer: Observer = None, weight: str=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._source = source
        self._target = target
        self._weight = weight
        self._observer = observer

    @staticmethod
    def create_and_append(source: Factor, target: Factor, observer: Observer, weight: str=None, tags=None, attributes=None):
        o = FactorsRelationDirectedFlowObservation(source, target, observer, weight, tags, attributes)
        if source:
            source.observations_append(o)
        if target:
            target.observations_append(o)
        if observer:
            observer.observables_append(source)
            observer.observables_append(target)
        return o

    @property
    def source_factor(self):
        return self._source

    @property
    def target_factor(self):
        return self._target

    @property
    def weight(self):
        return self._weight

    @property
    def observer(self):
        return self._observer

    @staticmethod
    def partial_key(source: Factor=None, target: Factor=None, observer: Observer=None):
        d = {"_t": RelationClassType.ff_directed_flow.name}
        if target:
            d["__t"] = target.ident

        if source:
            d["__s"] = source.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self):
        d = {"_t": RelationClassType.ff_directed_flow.name, "__s": self._source.ident, "__t": self._target.ident}
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
# PARAMETERS, BENCHMARKS, INDICATORS
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
        d = {
            'type': self._type,
            'range': self._range,
            'description': self._description,
            'default_value': self._default_value,
            'group': self._group,
            'current_value': self._current_value
        }

        d.update(Encodable.parents_encode(self, __class__))

        return d

    @staticmethod
    def partial_key(name: str=None):
        d = dict(_t="param")
        if name:
            d["_n"] = name
        return d

    def key(self):
        return {"_t": "param", "_n": self.name, "__o": self.ident}


class Benchmark(Encodable):
    """ A map from real values to a finite set of categories """
    def __init__(self, values, categories):
        self._values = values  # A list of sorted floats. Either ascending or descending
        self._categories = categories  # A list of tuples (label, color). One element less than the list of "values"

    def encode(self):
        d = {
            'values': self._values,
            'categories': self._categories
        }
        return d

    def category(self, v):
        # TODO Search in which category falls the numeric value "v", and return the corresponding tuple
        return None


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
    # TODO Expressions should support rich selectors, of the form "factors from processors matching these properties and factor types with those properties (include "all factors in a processor") AND/OR ..." plus set operations (union/intersection).
    # TODO Then, compute an operation over all selected factors.
    # TODO To generate multiple instances of the indicator or a single indicator accumulating many things.
    def __init__(self, name: str, formula: str, from_indicator: "Indicator", benchmark: Benchmark, indicator_category: IndicatorCategories, description=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        self._formula = formula
        self._from_indicator = from_indicator
        self._benchmark = benchmark
        self._indicator_category = indicator_category
        self._description = description

    def encode(self):
        d = {
            'formula': self._formula,
            'from_indicator': self._from_indicator,
            'benchmark': self._benchmark,
            'indicator_category': self._indicator_category.name,
            'description': self._description
        }

        d.update(Encodable.parents_encode(self, __class__))

        return d

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
