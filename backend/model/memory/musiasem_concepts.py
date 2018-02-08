# -*- coding: utf-8 -*-

"""
In-memory model

* Primitives
* Data structures
* Commands


Hierarchy / list
Graph (which can be a DAG)
Node
 Processor
 Factor
  Flow
  Fund
Observation
 Literal
 Expression. Solved expression. Specified / consequence.

Parameter
Indicator
Group (list of entities)  <<<
Geo
TimeExtent

ObservationProcess (Observer)
Source


=======
= RDF =
=======
The most flexible approach involves using RDF, RDFlib.
An interesting paper: "A survey of RDB to RDF translation approaches and tools"


"""

import collections

import json
from enum import Enum
from typing import *  # Type hints
import pint  # Physical Units management
import pandas as pd
import logging

from backend.common.helper import create_dictionary, strcmp, PartialRetrievalDictionary, case_sensitive
from backend.model import ureg
from backend.domain import State, get_case_study_registry_objects, LocallyUniqueIDManager
from backend.restful_service import log_level

logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# #################################################################################################################### #
# PRELIMINARY TYPES
# #################################################################################################################### #


class FlowFundRoegenType(Enum):  # Used in FlowFund
    flow = 1
    fund = 0


FactorInProcessorType = collections.namedtuple("FactorInProcessorType", "external incoming")

allowed_ff_types = ["int_in_flow", "int_in_fund", "int_out_flow", "ext_in_flow", "ext_out_flow"]

# #################################################################################################################### #
# BASE CLASSES
# #################################################################################################################### #


class HeterarchyNode:
    """ Taxon, Processor and Factor
        A hierarchy node can be a member of several hierarchies ¿what's the name of this property?
        A hierarchy can be flat, so a hierarchy node can be member of a simple list ¿heterarchy?
    """
    def __init__(self, parent=None, hierarchy=None):
        self._parent = {}
        self._children = {hierarchy: []}
        self.set_parent(parent, hierarchy)

    @property
    def parent(self):
        """ Return the parent. It works when the node is involved in at most one hierarchy """
        if len(self._parent) == 1:
            return self._parent[next(iter(self._parent))]
        elif len(self._parent) == 0:
            return None
        else:
            raise Exception("The node is involved in '"+str(len(self._parent))+"' hierarchies.")

    def get_parent(self, hierarchy=None):
        if hierarchy not in self._parent:
            return None
        else:
            return self._parent[hierarchy]

    def set_parent(self, p: "HeterarchyNode", hierarchy=None):
        # Check that parent has the same type
        if p and type(p) is not self.__class__:
            raise Exception("The hierarchy node class is '" + str(self.__class__) +
                            "' while the type of the parent is '" + str(type(p)) + "'.")

        self._parent[hierarchy] = p
        if p:
            if hierarchy not in p._children:
                p._children[hierarchy] = []
            p._children[hierarchy].append(self)

    def get_children(self, hierarchy=None):
        if hierarchy not in self._children:
            return None
        else:
            return self._children[hierarchy]

    @staticmethod
    def hierarchically_related(p1: "HeterarchyNode", p2: "HeterarchyNode", h: "Heterarchy" =None):
        return p1.get_parent(h) == p2 or p2.get_parent(h) == p1

# Connection = collections.namedtuple("Connection", "source destination hierarchical weight")
#
#
# class Connectable:
#     """ Connect entities of the same type
#         For Processor and Factor (Factor=a FactorType in a Processor)
#         Two types of connection: hierarchy (parent-child) and side
#         (direct connection, no matter which hierarchical connection there exists)
#     """
#     # TODO Possibility of having different "ConnectionSet"s
#     # TODO Similar to the possibility of HierarchyNodes being members of several hierarchies
#     # TODO In this case, there would be a base ConnectionSet (None), then additional ones would be
#     # TODO for options. A ConnectionSet could have tags and attributes, so it would be possible to filter
#     def __init__(self):
#         self._connections = []  # type: Connection
#
#     @property
#     def connections(self):
#         return self._connections
#
#     def connect_to(self, other_entity_same_type: "Connectable", h: "Heterarchy" =None, weight: float=1.0):
#         return Connectable.connect(self, other_entity_same_type, h, weight)
#
#     def connect_from(self, other_entity_same_type: "Connectable", h: "Heterarchy" =None, weight: float=1.0):
#         return Connectable.connect(other_entity_same_type, self, h, weight)
#
#     def generate_expression(self):
#         # TODO If it is a factor, generate expression relating the factors with all the input or (not and) output connections
#         pass
#
#     @staticmethod
#     def connect(entity: "Connectable", other_entity_same_type: "Connectable", h: "Heterarchy", weight: float):
#         if other_entity_same_type and type(other_entity_same_type) is not entity.__class__:
#             raise Exception("The source class is '" + str(entity.__class__) +
#                             "' while the destination type is '" + str(type(other_entity_same_type)) + "'.")
#
#         # TODO Register connections? Good for elaboration of constraints
#         # TODO If self is FACTOR, check that the direction is compatible with the FactorInProcessorType
#         # TODO   If NOT, ¿overwrite? ¿issue an error?
#         # Hierarchical or sequential connection?
#         if isinstance(entity, Factor):
#             p1 = entity._processor
#             p2 = other_entity_same_type._processor
#         else:
#             p1 = entity
#             p2 = other_entity_same_type
#
#         hierarchical = HeterarchyNode.hierarchically_related(p1, p2, h)
#         # TODO If hierarchical==false, check that there is no hierarchical relationship, like grandparent-grandchild or longer
#         c = Connection(source=entity, destination=other_entity_same_type, hierarchical=hierarchical, weight=weight)
#         entity._connections.append(c)
#         other_entity_same_type._connections.append(c)
#         return c


class Identifiable:
    def __init__(self):
        self._id = LocallyUniqueIDManager().get_new_id()

    @property
    def ident(self):
        return self._id


class Nameable:
    """ Entities with name. Almost all. They contain the name used in the registry. """
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n: str):
        self._name = n


class Taggable:
    def __init__(self, tags):
        self._tags = []
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

        self._tags.extend(lst)


class Automatable:
    """ An entity that could have been generated by an automatic process. A flag, the producer object and a reason. """
    def __init__(self):
        self._automatically_generated = False
        self._producer = None  # Object (instance) responsible of the production
        self._generation_reason = None  # Clone, automatic reasoning, solving

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


class Qualifiable:
    """ An entity with Attributes """
    def __init__(self, attributes=None):
        self._attributes = create_dictionary()
        if attributes:
            for k in attributes:
                self.attributes_append(k, attributes[k])

    @property
    def attributes(self):
        return self._attributes

    def attributes_append(self, name, value):
        self._attributes[name] = value


class Observable:
    """ An entity which can be structurally or quantitatively observed.
        It can have none to infinite possible observations
    """
    def __init__(self, location: "Geolocation"):
        self._location = location  # Definition of where the observable is
        self._physical_nature = None
        self._observations = []  # type: List[Union[FactorQuantitativeObservation, ProcessorObservation]]

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

# #################################################################################################################### #
# VALUE OBJECTS for SPACE and TIME
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
    """ The base class for quantitative observations. For structural observations (one of many structures are possible,
        the initial proposal is an Expression
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

# #################################################################################################################### #
# HETERARCHIES and TAXA
# #################################################################################################################### #


class Heterarchy(Nameable):
    """ A list or a taxonomy, made of "Taxon" instances """
    def __init__(self, name=None, roots=None):
        Nameable.__init__(self, name)
        self._roots = []  # List of root "IHierarchyNode" nodes (the object serves to represent a list)
        self._type = None # The hierarchy should be of a single type. The first element of the hierarchy sets the type, and new elements must be of the same type
        self._level_names = None # Each level of the hierarchy can have a name. This list register these names, from root to leaves
        if roots:
            self.roots_append(roots)

    @property
    def roots(self):
        return self._roots

    def roots_append(self, root):
        if isinstance(root, (list, set)):
            first = root[0]
            lst = root
        else:
            first = root
            lst = [root]

        if not self._type:
            self._type = first

        self._roots.extend(lst)

    @property
    def level_names(self):
        return self._level_names

    @level_names.setter
    def level_names(self, lvls: list):
        self._level_names = lvls

    def get_node(self, name) -> HeterarchyNode:
        """ Find a node of the hierarchy named "name" """
        def recursive_get_node(lst):
            for n in lst:
                if n.name.lower() == name.lower():
                    return n

            for n in lst:
                if n.get_children(self):
                    f = recursive_get_node(n.get_children(self))
                    if f:
                        return f
            return None

        return recursive_get_node(self._roots)


class HierarchyExpression:
    def __init__(self, expression: QualifiedQuantityExpression=None):
        # Defines how to compute a HeterarchyNode relative to other HeterarchyNodes
        # The hierarchical relation implies a parent = SUM children expression. If specified, this implicit relation
        # would be overriden
        self._expression = expression

    @property
    def expression(self):
        # Get expression defining
        return self._expression

    @expression.setter
    def expression(self, e: QualifiedQuantityExpression=None):
        self._expression = e


class Taxon(Identifiable, Nameable, HeterarchyNode, HierarchyExpression):
    def __init__(self, name, parent=None, hierarchy=None, expression=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        HeterarchyNode.__init__(self, parent, hierarchy)
        HierarchyExpression.__init__(self, expression)

    def key(self):
        return {"_t": "t", "_n": self.name, "__id": self.ident}


class HierarchiesSet(Nameable):
    """ A set of Hierachies """
    def __init__(self, name):
        Nameable.__init__(name)
        self._hs = create_dictionary()

    def append(self, h: str, member: Union[str, List[str]]):
        # Look for hierarchy "h"
        if h not in self._hs:
            self._hs[h] = Heterarchy(h)
        # Insert "member" in hierarchy "h"
        self._hs[h].roots_append(member)

    def search(self, h: str, member: str):
        """ Returns the hierarchy/ies containing "member" """
        lst = []
        if not h and member:  # Hierarchy "h" not specified, look for "member" in ALL hierarchies
            for h2 in self._hs:
                if self._hs[h2].get_node(member):
                    lst.append(h2)
        elif h and not member:  # Only hierarchy "h" specified, return "h" if it exists
            if h in self._hs:
                lst.append(h)
        else:
            if h in self._hs and self._hs[h].get_node(member):
                lst.append(h)

        return lst

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
    def partial_key(name: str, registry: PartialRetrievalDictionary):
        if name:
            return {"_t": "o", "_n": name}
        else:
            return {"_t": "o"}

    def key(self, registry: PartialRetrievalDictionary):
        """
        Return a Key for the identification of the Observer in the registry
        :param registry:
        :return:
        """
        return {"_t": "o", "_n": self.name, "__id": self.ident}


class FactorType(Identifiable, Nameable, HeterarchyNode, HierarchyExpression, Taggable, Qualifiable):  # Flow or fund type (not attached to a Processor)
    """ A Factor as type, in a hierarchy, a Taxonomy """
    def __init__(self, name, parent=None, hierarchy=None,
                 tipe: FlowFundRoegenType=FlowFundRoegenType.flow,
                 tags=None, attributes=None, expression=None):
        Identifiable.__init__(self)
        Nameable.__init__(self, name)
        HeterarchyNode.__init__(self, parent, hierarchy)
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        HierarchyExpression.__init__(self, expression)
        self._roegen_type = tipe
        self._physical_type = None  # TODO Which physical types. An object
        self._default_unit_str = None  # TODO A string representing the unit, compatible with the physical type
        self._factors = []

    def full_hierarchy_name(self, hierarchy=None):
        """
        Obtain the full hierarchy name of the current FactorType

        :return:
        """
        p = self
        lst = []
        while p:
            lst.insert(0, p.name)
            par = getattr(p, "get_parent", None)
            if par:
                p = p.get_parent(hierarchy)
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

    @staticmethod
    def partial_key(name: str, registry: PartialRetrievalDictionary=None):
        if name:
            return {"_t": "ft", "_n": name}
        else:
            return {"_t": "ft"}

    def key(self, registry: PartialRetrievalDictionary=None):
        return {"_t": "ft", "_n": self.name, "__id": self.ident}


class Processor(Identifiable, Nameable, Taggable, Qualifiable, Automatable, Observable):
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
        self._type = None
        self._external = external  # Either external (True) or internal (False)

        # The processor references a previously defined processor
        # * If a factor is defined, do not look in the referenced
        # * If a factor is not defined here, look in the referenced
        # * Other properties can also be referenced (if defined locally it is not searched in the referenced)
        # * FactorObservations are also assimilated
        #   (although expressions evaluated in the context of the local processor, not the referenced one)
        # * Relations are NOT assimilated, at least by default. THIS HAS TO BE STUDIED IN DEPTH
        self._referenced_processor = referenced_processor

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

    def full_hierarchy_names(self, registry: PartialRetrievalDictionary):
        """
        Obtain the full hierarchy name of the current processor
        It looks for the PART-OF relations in which the processor is in the child side

        :param registry:
        :return:
        """
        # Get matching relations
        part_of_relations = registry.get(ProcessorsRelationPartOfObservation.partial_key(child=self, registry=registry))

        # Compose the name, recursively
        if len(part_of_relations) == 0:
            return [self.name]
        else:
            # Take last part of the name
            last_part = self.name.split(".")[-1]
            return [(p+"."+last_part) for rel in part_of_relations for p in rel.parent_processor.full_hierarchy_names(registry)]

    def clone(self, state: State, p_set: "ProcessorsSet", reference=True, objects_processed: dict= {}):
        glb_idx, _, _, _, _ = get_case_study_registry_objects(state)
        if reference:
            p = Processor(self.name, referenced_processor=self)
            # TODO Find relations in which the processor is parent, to copy children
            # TODO Clone these processors and the relations
            part_of_relations = glb_idx.get(ProcessorsRelationPartOfObservation.partial_key(parent=self, registry=glb_idx))
            upscale_relations = glb_idx.get(ProcessorsRelationUpscaleObservation.partial_key(parent=self, registry=glb_idx))
            for rel in part_of_relations + upscale_relations:
                if rel.child not in objects_processed:
                    p2 = rel.child.clone(state, p_set, reference, objects_processed)
                    objects_processed[rel.child] = p2
                else:
                    p2 = objects_processed[rel.child]

                if isinstance(rel, ProcessorsRelationPartOfObservation):
                    o1 = ProcessorsRelationPartOfObservation.create_and_append(p, p2, rel.observer)
                    glb_idx.put(o1.key(glb_idx), o1)
                else:  # Upscale relation
                    if rel.target_processor not in objects_processed:
                        pass
                rel.clone(state)
            # TODO Flow relations do not clone the other, ONLY the relation
            flow_relations = glb_idx.get(ProcessorsRelationUndirectedFlowObservation.partial_key(source=self, registry=glb_idx))
        else:
            p = Processor(self.name,
                          external=self.external, location=self._location,
                          tags=self._tags, attributes=self._attributes,
                          referenced_processor=self.referenced_processor
                          )
            # Clone factors also
            for f in self.factors:
                f_ = Factor.clone(f, p)
                p.factors_append(f_)

        p_set.append(p)
        return p

    @staticmethod
    def alias_key(name: str, processor: "Processor", registry: PartialRetrievalDictionary=None):
        """
        Creates a registrable entry to allow a different name for "processor"
        Key TO Key
        :param name:
        :param processor:
        :param registry:
        :return: A tuple formed by the key and value to be registered
        """
        return {"_t": "p", "_n": name, "__id": processor.ident, "_aka": True}

    @staticmethod
    def is_alias_key(composite_key: dict):
        return "_aka" in composite_key

    @staticmethod
    def partial_key(name: str, registry: PartialRetrievalDictionary=None):
        if name:
            return {"_t": "p", "_n": name}
        else:
            return {"_t": "p"}

    def key(self, registry: PartialRetrievalDictionary=None):
        """
        Return a Key for the identification of the Processor in the registry
        :param registry:
        :return:
        """
        return {"_t": "p", "_n": self.name, "__id": self.ident}


class Factor(Identifiable, Nameable, Taggable, Qualifiable, Observable, Automatable):
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

    @property
    def processor(self):
        return self._processor

    @property
    def referenced_factor(self):
        return self._referenced_factor

    @staticmethod
    def create(name, processor: Processor, in_processor_type: FactorInProcessorType, taxon: FactorType,
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
    def clone(factor: "Factor", processor: Processor):
        f = Factor.create(factor.name, processor, in_processor_type=factor.type, taxon=factor._taxon,
                          location=factor._location, tags=factor._tags, attributes=factor._attributes)
        # Observations
        for o in f.observations:
            o_ = o.clone()

    @staticmethod
    def partial_key(processor: Processor=None, taxon: FactorType=None, registry: PartialRetrievalDictionary=None):
        d = {"_t": "f"}
        if processor:
            d["__p"] = processor.ident
        if taxon:
            d["__ft"] = taxon.ident

        return d

    def key(self, registry: PartialRetrievalDictionary):
        """
        Return a Key for the identification of the Factor in the registry
        :param registry:
        :return:
        """
        return {"_t": "f", "__id": self.ident, "__p": self._processor.ident, "__ft": self._taxon.ident}


class ProcessorsSet(Nameable):
    """ A set of Processors """
    def __init__(self, name):
        Nameable.__init__(self, name)
        self._pl = set([])  # Processors members of the set
        self._attributes = {}  # Attributes characterizing the processors. {name: code list}

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
        with the goal of elaborating a code list for each attribute

        FOR NOW THIS HAS NO APPLICATION, IT IS AN INDEX TO THIS INFORMATION IN A DIFFERENT WAY

        :param d:
        :return:
        """
        for k in d:
            if k not in self._attributes:
                s = set()
                self._attributes = s
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


####################################################################################################
# NOT USABLE RIGHT NOW!!!!!!!!!!!!!!!!!!!!!!!!!
####################################################################################################
# def connect_processors(source_p: Processor, dest_p: Processor, h: "Heterarchy", weight: float, taxon: FactorType, source_name: str=None, dest_name: str=None):
#     """ High level function to connect two processors adding factors if necessary """
#     if not dest_name and source_name:
#         dest_name = source_name
#     hierarchical = HeterarchyNode.hierarchically_related(source_p, dest_p, h)
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
    ff_directed_flow = 4
    ff_reverse_directed_flow = 5


class FactorQuantitativeObservation(Taggable, Qualifiable, Automatable):
    """ An expression or quantity assigned to an Observable (Factor) """
    def __init__(self, v: QualifiedQuantityExpression, factor: Factor=None, observer: Observer=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Automatable.__init__(self)
        self._value = v
        self._factor = factor
        self._observer = observer

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
    def partial_key(factor: Factor=None, observer: Observer=None, registry: PartialRetrievalDictionary=None):
        d = {"_t": "qq"}
        if factor:
            d["__f"] = factor.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self, registry: PartialRetrievalDictionary=None):
        return {"_t": "qq", "__f": self._factor.ident, "__oer": self._observer.ident}


class RelationObservation(Taggable, Qualifiable, Automatable):  # All relation kinds
    pass


class ProcessorsRelationObservation(RelationObservation):  # Just base of ProcessorRelations
    pass


class FactorsRelationObservation(RelationObservation):  # Just base of FactorRelations
    pass


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
    def partial_key(parent: Processor=None, child: Processor=None, observer: Observer=None, registry: PartialRetrievalDictionary=None):
        d = {"_t": RelationClassType.pp_part_of.name}
        if child:
            d["__c"] = child.ident

        if parent:
            d["__p"] = parent.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self, registry: PartialRetrievalDictionary=None):
        return {"_t": RelationClassType.pp_part_of.name, "__p": self._parent.ident, "__c": self._child.ident, "__oer": self._observer.ident}


class ProcessorsRelationUndirectedFlowObservation(ProcessorsRelationObservation):
    """
    Represents an undirected Flow, from a source to a target Processor
    Undirected flow is DYNAMICALLY converted to Directed flow, for each factor:
    * If the factor of the source (parent) is "Incoming", the flow is from the parent to the child
    * If the factor of the source (parent) is "Outgoing", the flow is from the child to the parent

    from backend.model.memory.musiasem_concepts import Processor, Observer, ProcessorsRelationUndirectedFlowObservation
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
    def partial_key(source: Processor=None, target: Processor=None, observer: Observer=None, registry: PartialRetrievalDictionary=None):
        d = {"_t": RelationClassType.pp_undirected_flow.name}
        if target:
            d["__t"] = target.ident

        if source:
            d["__s"] = source.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self, registry: PartialRetrievalDictionary=None):
        return {"_t": RelationClassType.pp_undirected_flow.name, "__s": self._source.ident, "__t": self._target.parent, "__oer": self._observer.ident}


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
        fo = FactorQuantitativeObservation.create_and_append(v=QualifiedQuantityExpression(parent_name + "." + self._factor_name+" * ("+self._quantity+")"),
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
    def partial_key(parent: Processor=None, child: Processor=None, observer: Observer=None, registry: PartialRetrievalDictionary=None):
        d = {"_t": RelationClassType.pp_upscale.name}
        if child:
            d["__c"] = child.ident

        if parent:
            d["__p"] = parent.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self, registry: PartialRetrievalDictionary=None):
        return {"_t": RelationClassType.pp_upscale.name, "__p": self._parent.ident, "__c": self._child.ident, "__oer": self._observer.ident}


class FactorsRelationDirectedFlowObservation(FactorsRelationObservation):
    """
    Represents a directed Flow, from a source to a target Factor
    from backend.model.memory.musiasem_concepts import Processor, Factor, Observer, FactorsRelationDirectedFlowObservation
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
    def observer(self):
        return self._observer

    @staticmethod
    def partial_key(source: Factor=None, target: Factor=None, observer: Observer=None, registry: PartialRetrievalDictionary=None):
        d = {"_t": RelationClassType.ff_directed_flow.name}
        if target:
            d["__t"] = target.ident

        if source:
            d["__s"] = source.ident

        if observer:
            d["__oer"] = observer.ident

        return d

    def key(self, registry: PartialRetrievalDictionary=None):
        return {"_t": RelationClassType.ff_directed_flow.name, "__s": self._source.ident, "__t": self._target.ident, "__oer": self._observer.ident}

# #################################################################################################################### #
# NUSAP Pedigree
# #################################################################################################################### #


class PedigreeTemplate(Nameable):
    """ A pedigree template (to represent a Pedigree Matrix), made of a dictionary of lists """
    def __init__(self, name):
        Nameable.__init__(self, name)
        self._categories = create_dictionary()


class Pedigree:
    pass

# #################################################################################################################### #
# PARAMETERS, BENCHMARKS, INDICATORS
# #################################################################################################################### #


class Parameter(Nameable):
    """ A numeric variable changeable directly by an analyst """
    def __init__(self, name):
        Nameable.__init__(self, name)
        self._range = None
        self._description = None
        self._type = None
        self._current_value = None
        self._default_value = None


class Benchmark:
    """ A map from real values to a finite set of categories """
    def __init__(self, values, categories):
        self._values = values  # A list of sorted floats. Either ascending or descending
        self._categories = categories  # A list of tuples (label, color). One element less than the list of "values"

    def category(self, v):
        # TODO Search in which category falls the numeric value "v", and return the corresponding tuple
        return None


class Indicator:
    """ A numeric value result of operating several observations
    """
    # TODO Support for a wildcard indicator. Compute the same operation over all factors of a processor, over all processors.
    # TODO To generate multiple instances of the indicator or a single indicator acumulating many things.
    def __init__(self):
        self._indicator = None

# #################################################################################################################### #
# MAPPING, EXTERNAL DATASETS
# #################################################################################################################### #


class Mapping:
    """
    Transcription of information specified by a mapping command
    """
    def __init__(self, name, source, dataset, origin, destination, the_map: List[Tuple]):
        self.name = name
        self.source = source
        self.dataset = dataset
        self.origin = origin  # Dimension
        self.destination = destination # Destination Dimension
        self.map = the_map  # List of tuples (pairs) source code, destination code[, expression]


# TODO Currently ExternalDataset is not used. Dataset (which can be persisted) is the current choice
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
