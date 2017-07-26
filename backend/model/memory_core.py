# -*- coding: utf-8 -*-

"""
In-memory model

* Primitives
* Data structures
* Commands

Store <<<
CaseStudy <<<
Submission <<<
Transaction <<<
Command <<<
User <<<

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

"""

import collections


from typing import * # Type hints
import abc # Abstract Base Class
from enum import Enum
import anytree  # Entities member of more than one tree
import pint # Physical Units management
import aadict
from decimal import * # Use of exact operations
import json

import cubes

from ..helper import create_dictionary
from . import ureg

# #################################################################################################################### #


class FlowFundRoegenType(Enum):  # Used in FlowFund
    flow = 1
    fund = 0

FactorInProcessorType = collections.namedtuple("FactorInProcessorType", "external incoming")

# #################################################################################################################### #
# BASE CLASSES


class HierarchyNode:
    """ Taxon, Processor and Factor
        A hierarchy node can be a member of several hierarchies
        A hierarchy can be flat, so a hierarchy node can be member of a simple list
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

    def set_parent(self, p: "HierarchyNode", hierarchy=None):
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
    def hierchically_related(p1: "HierarchyNode", p2: "HierarchyNode", h: "Hierarchy"=None):
        return p1.get_parent(h) == p2 or p2.get_parent(h) == p1

Connection = collections.namedtuple("Connection", "source destination hierarchical weight")


class Connectable:
    """ Connect entities of the same type
        For Processor and Factor (a FactorTaxon in a Processor)
        Two types of connection: hierarchy (parent-child) and side
        (direct connection, no matter which hierarchical connection there exists)
    """
    # TODO Possibility of having different "ConnectionSet"s
    # TODO Similar to the possibility of HierarchyNodes being members of several hierarchies
    # TODO In this case, there would be a base ConnectionSet (None), then additional ones would be
    # TODO for options. A ConnectionSet could have tags and attributes, so it would be possible to filter
    def __init__(self):
        self._connections = []  # Connection (namedtuple)

    @property
    def connections(self):
        return self._connections

    def connect_to(self, other_entity_same_type: "Connectable", h: "Hierarchy"=None, weight: float=1.0):
        return Connectable.connect(self, other_entity_same_type, h, weight)

    def connect_from(self, other_entity_same_type: "Connectable", h: "Hierarchy"=None, weight: float=1.0):
        return Connectable.connect(other_entity_same_type, self, h, weight)

    def generate_expression(self):
        # TODO If it is a factor, generate expression relating the factors with all the input or (not and) output connections
        pass

    @staticmethod
    def connect(entity: "Connectable", other_entity_same_type: "Connectable", h: "Hierarchy", weight: float):
        if other_entity_same_type and type(other_entity_same_type) is not entity.__class__:
            raise Exception("The source class is '" + str(entity.__class__) +
                            "' while the destination type is '" + str(type(other_entity_same_type)) + "'.")

        # TODO Register connections? Good for elaboration of constraints
        # TODO If self is FACTOR, check that the direction is compatible with the FactorInProcessorType
        # TODO   If NOT, ¿overwrite? ¿issue an error?
        # Hierarchical or sequential connection?
        if isinstance(entity, Factor):
            p1 = entity._processor
            p2 = other_entity_same_type._processor
        else:
            p1 = entity
            p2 = other_entity_same_type

        hierarchical = HierarchyNode.hierchically_related(p1, p2, h)
        # TODO If hierarchical==false, check that there is no hierarchical relationship, like grandparent-grandchild or longer
        c = Connection(source=entity, destination=other_entity_same_type, hierarchical=hierarchical, weight=weight)
        entity._connections.append(c)
        other_entity_same_type._connections.append(c)
        return c


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
    """ An entity that could have generated by an automatic process. A flag and a reason. """
    def __init__(self):
        self._automatically_generated = False
        self._generation_reason = None  # Clone, automatic reasoning, solving

    @property
    def automatically_generated(self):
        return self._automatically_generated

    @automatically_generated.setter
    def automatically_generated(self, aut: bool):
        self._automatically_generated = aut

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

# #################################################################################################################### #
# Entities


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
        return QualifiedQuantityExpression(json.dumps({'n': n}))

    @staticmethod
    def nu(n: float, u: str):
        # Check that "u" is a recognized unit type
        try:
            ureg(u)
        except pint.errors.UndefinedUnitError:
            # The user should know that the specified unit is not recognized
            raise Exception("The specified unit '" + u + "' is not recognized")
        return QualifiedQuantityExpression(json.dumps({'n': n, 'u': u}))


class Source(Qualifiable):
    """ A Dataset or any source of information in general """
    def __init__(self, attributes):
        attributes2 = attributes.clone()
        # TODO If present, remove the following attributes from the dictionary
        self._description = None  # What is the content of the source
        self._location = None  # Where is the source accessible
        self._address = None  # Address to have direct access to the source
        Qualifiable.__init__(attributes2)


class Observer(Nameable):
    """ An entity capable of producing Observations on an Observable. It is equivalent to a process """
    def __init__(self, name, description=None):
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
                self._observables = o

    @property
    def sources(self):
        return self._sources

    def sources_append(self, source):
        if isinstance(source, (list, set)):
            lst = source
        else:
            lst = [source]

        self._sources.extend(lst)


class Observable:
    """ An entity representing a measurable aspect, something which can be observed, accounted. 
       It can have none to infinite possible observations
    """
    def __init__(self, location: "Geolocation"):
        self._location = location  # Definition of where the observable is
        self._physical_nature = None
        self._observations = []  # type: List[FactorObservation]

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


class ConnectionsSet(Nameable):
    """ A set of connections. It allows having different connection schemes for the processors+factors network """
    def __init__(self, name):
        self._name = name
        self._connections = None

    @property
    def connections(self):
        return self._connections

    @connections.setter
    def connections_append(self, connection):
        if isinstance(connection, (list, set)):
            lst = connection
        else:
            lst = [connection]

        self._connections.extend(lst)


class ProcessorsSet(Nameable):
    """ A set of Processors """
    def __init__(self, name):
        Nameable.__init__(name)

    def clone(self):
        pass
    # TODO Manage the set


class Hierarchy(Nameable):
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

    def get_node(self, name) -> HierarchyNode:
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
        # Defines how to compute a HierarchyNode relative to other HierarchyNodes
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


class Taxon(Nameable, HierarchyNode, HierarchyExpression):
    def __init__(self, name, parent=None, hierarchy=None, expression=None):
        Nameable.__init__(self, name)
        HierarchyNode.__init__(self, parent, hierarchy)
        HierarchyExpression.__init__(self, expression)


class Processor(Nameable, HierarchyNode, Taggable, Qualifiable, Connectable, Automatable):
    def __init__(self, name, parent=None, hierarchy=None, external: bool=False, tags=None, attributes=None):
        Nameable.__init__(self, name)
        HierarchyNode.__init__(self, parent, hierarchy)
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Connectable.__init__(self)
        Automatable.__init__(self)
        self._factors = []  # type: List[Factor]
        self._type = None
        self._external = external  # Either external (True) or internal (False)

    @property
    def factors(self):
        return self._factors

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
        return self._external

    @property
    def internal(self):
        return not self._internal

    @property
    def type_processor(self):
        # TODO True if the processor is a type. The alternative is the processor being a real. The type abstract a function
        # TODO "type" and functional is similar. But it depends on the analyst, who can also define the processor as
        # TODO structural and "type" at the same time
        return False

    @type_processor.setter
    def type_processor(self, v: bool):
        self._type = v

    @property
    def real_processor(self):
        return not self.type_processor

    @real_processor.setter
    def real_processor(self, v: bool):
        self.type_processor = not v


class FactorTaxon(Nameable, HierarchyNode, HierarchyExpression, Taggable, Qualifiable):  # Flow or fund type (not attached to a Processor)
    """ A Taxon in a hierarchy, a Taxonomy """
    def __init__(self, name, parent=None, hierarchy=None,
                 tipe: FlowFundRoegenType=FlowFundRoegenType.flow,
                 tags=None, attributes=None, expression=None):
        Nameable.__init__(self, name)
        HierarchyNode.__init__(self, parent, hierarchy)
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        HierarchyExpression.__init__(self, expression)
        self._roegen_type = tipe
        self._physical_type = None  # TODO Which physical types. An object
        self._default_unit_str = None  # TODO A string representing the unit, compatible with the physical type
        self._factors = []

    # A registry of all factors refering to this FactorTaxon
    @property
    def factors(self):
        return self._factors

    def factors_append(self, factor: "Factor"):
        self._factors.append(factor)

    @property
    def roegen_type(self):
        return self._roegen_type


class Factor(Nameable, Taggable, Qualifiable, Connectable, Observable, Automatable):
    """ A Flow or Fund when attached to a Processor """
    def __init__(self, name, processor: Processor, in_processor_type: FactorInProcessorType, taxon: FactorTaxon,
                 location: "Geolocation"=None, tags=None, attributes=None):
        Nameable.__init__(self, name)
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        Connectable.__init__(self)
        Observable.__init__(self, location)
        Automatable.__init__(self)
        self._processor = processor
        self._taxon = taxon
        self._type = in_processor_type

    @property
    def processor(self):
        return self._processor

    @staticmethod
    def create(name, processor: Processor, in_processor_type: FactorInProcessorType, taxon: FactorTaxon,
                 location: "Geolocation"=None, tags=None, attributes=None):
        f = Factor(name, processor, in_processor_type, taxon, location, tags, attributes)
        if processor:
            processor.factors_append(f)
        if taxon:
            taxon.factors_append(f)
        return f

    @property
    def type(self):  # External (or internal), Incoming (or outgoing)
        return self._type

    @property
    def roegen_type(self):
        return self._taxon.roegen_type


def connect_processors(source_p: Processor, dest_p: Processor, h: "Hierarchy", weight: float, taxon: FactorTaxon, source_name: str=None, dest_name: str=None):
    """ High level function to connect two processors adding factors if necessary """
    if not dest_name and source_name:
        dest_name = source_name
    hierarchical = HierarchyNode.hierchically_related(source_p, dest_p, h)
    f1 = None
    f2 = None
    if source_name:
        # Find a factor with source_name in source_p
        for f in source_p.factors:
            if source_name.lower() == f.name:
                f1 = f
                break
        # If found, the type should be congruent. If it is an hierarchical connection, the sense is
        # reversed (IN allows a connection OUT to its children, which must be also IN)
        if f1:
            # TODO Check all possibilities with tests (in hierarchical, out hierarchical, in sequential, out sequential)
            ok = (not f1.type.incoming and not hierarchical) or (f1.type.incoming and hierarchical)
            if not ok:
                raise Exception("A factor by the name '"+source_name+"' already exists for source Processor "
                                "'"+source_p.name+"' and the sense of connection is not congruent with connection "
                                "to be added")
    if dest_name:
        # Find a factor with dest_name in dest_p
        for f in dest_p.factors:
            if dest_name.lower() == f.name:
                f2 = f
                break
        # If found, the type should be congruent. If it is an hierarchical connection, the sense is
        # reversed (IN allows a connection OUT to its children, which must be also IN)
        if f2:
            # TODO Check all possibilities with tests (in hierarchical, out hierarchical, in sequential, out sequential)
            ok = (f2.type.incoming and not hierarchical) or (not f2.type.incoming and hierarchical)
            if not ok:
                raise Exception("A factor by the name '"+source_name+"' already exists for destination Processor "
                                "'"+dest_p.name+"' and the sense of connection is not congruent with connection "
                                "to be added")

    if not f1:
        if not hierarchical:
            incoming = False
        else:
            incoming = True
        f1 = Factor.create(source_name, source_p, FactorInProcessorType(external=source_p.external, incoming=incoming), taxon)

    if not f2:
        if not hierarchical:
            incoming = True
        else:
            incoming = False
        f2 = Factor.create(dest_name, dest_p, FactorInProcessorType(external=dest_p.external, incoming=incoming), taxon)

    c = f1.connect_to(f2, h, weight)

    return c, f1, f2


# #################################################################################################################### #


class Geolocation:
    """ For the geolocation of processors. Factors and Observations could also be qualified with Geolocation """
    def __init__(self, name, projection=None, shape=None):
        self._name = name
        self._projection = projection
        self._shape = shape

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

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


# #################################################################################################################### #
# NUSAP Pedigree

class PedigreeTemplate(Nameable):
    """ A pedigree template (to represent a Pedigree Matrix), made of a list of lists """
    def __init__(self):
        self._


class Pedigree:
    pass

# #################################################################################################################### #


class FactorObservation(Taggable, Qualifiable):
    """ An expression or quantity assigned to an Observable (Factor) """
    def __init__(self, v: QualifiedQuantityExpression, factor: Factor=None, observer: Observer=None, tags=None, attributes=None):
        Taggable.__init__(self, tags)
        Qualifiable.__init__(self, attributes)
        self._value = v
        self._factor = factor
        self._observer = observer

    @staticmethod
    def create(v, factor: Factor, observer: Observer, tags=None, attributes=None):
        o = FactorObservation(v, factor, observer, tags, attributes)
        if factor:
            factor.observations_append(o)
        if observer:
            observer.observables_append(factor)

    @property
    def factor(self):
        return self._factor

    @property
    def observer(self):
        return self._observer

    @property
    def value(self):
        return self._value


class Parameter:
    """ A numeric variable changeable directly by an analyst """
    def __init__(self):
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
        self._
