"""
Model for MuSIASEM

Elements
* Case studies
* Processors
* Observations on Processors or sets of Processors
* Factors (flows and funds), which are Observations
* Hierarchies. Can be typologies "is-a" or structural "part-of"
* Links between factors (of the same type).

Use cases
* Processors containing flows and funds (flow or fund are "factor")
* Factors can be extensive or intensive. To express an intensive, in the observation put the factor to which the relation can be transformed into extensive
* Factors can be observed by different Observers 
* Allow several observations (pair Observable-Observer).
  * Clear separation of top-down and bottom-up (how?)
    * Top-down can be directly specified
    * Bottom-up is an expression.
* Observations of factors can be deterministic or probabilistic (NUSAP)
  * It can be an expression (a template processor)
* The observation has a provenance specified by the Observer
* There are KPI's (like Bioeconomic Pressure)
* Processors can be TYPES (aggregating real processors according to some criteria) or REAL (specific processor, that can also be the expression of a function -virtual-)
* Processors can contain other processors as specializations and/or part-of
* Processors can be extensive or intensive if all its factors are such
* Losses. Either a structural processor with no output, or an input to "Eco" processor to accumulate
* Processors are linked to other processors
  * Same level
  * Parent-child
  * NO distinction between same level and parent-child connections
* Factors can contain other factors as "is-a"
* Factor hierarchies. There can exist transformations between factor hierarchies
* Allow building MuSIASEM case studies in multiple steps. For instance, upscaling
* Observations (factors and KPIs) can have zero or more benchmarks
* Link factors
  * How to represent a bundle of links?? Expression?? Wildcard?? (like "all factors which are leaf nodes in the typology T")
* Factors by default inherit the hierarchies of the Processors in which they are embedded
  * Parent-child links created by default
  * Lateral links can change the previous
 
Case studies
* Almería. Upscaling of intensive processors. Conversion to extensive
* Soslaires. All extensive processors related both hierarchically and sequentially
* Energy (Maddalena). Tree structure of Processors
* Michele Food case study.
"""
from enum import Enum

"""
Observable. Can be a flow, a fund or other, like KPI calculations
Observer.
Observation: Extensive. Intensive. Formula. Hierarchy of classes, the first would be NUSAP

"""


class Node:
    """
    Member of one or more hierarchies
    """
    def __init__(self, name):
        self.name = name
        self.hierarchies = dict() # Which hierarchies the node is member of


class ProcessorDetails:
    """ Contains a list of Observables """
    def __init__(self):
        pass


class ProcessorType(Enum):
    type = 1
    real = 2


class Processor(Node):
    def __init__(self):
        pass


class Observable(Node):
    """
    Something numeric or qualitative about a processor
    The observable has the following attributes
    * What it is. A flow, a fund, a KPI. A class of Observable (can be member of hierarchies)
    * What it can be. Extensive, intensive (relative to other observable), literal, formula
    """
    def __init__(self):
        pass


class Factor(Observable):
    """ Flow or fund """


class Fund(Factor):
    """ """


class Observer:
    """
    An entity capable of issuing Observations and explaining how they have been obtained
    """
    def __init__(self):
        pass


class Observation:
    """
    An observation about an Observable made by an Observer
    It can be a qualified number or a formula
    """
    def __init__(self, observable: Observable, observer: Observer):
        pass


class HierarchyType:
    is_a = 1
    part_of = 2


class HierarchyNode:
    def __init__(self, parent, name, node: Node=None, h: Hierarchy=None):
        self.name = name
        self.node = node # The payload is not in the hierarchy itself. The hierarchy is just the structure
        self.hierarchy_dependant_node_details = None
        self.children = []
        self.set_parent(parent)
        self.type = HierarchyType.part_of
        self.hierarchy = h
        if self.hierarchy is None:
            # TODO Find hierarchy of ancestor and assume the same
            c = self.parent

    def add_children(self, children: list, overwrite=False):
        """ Add children to the current HierarchyNode. This allows the composition of a hierarchy
        """
        # Check that all "children" are of the same class of the current node
        for c in children:
            if type(c) is not self.__class__:
                raise Exception("All children should be instances of '" + str(self.__class__) +
                                "'. Instance of '" + str(type(c)) + "' found.")

        # Change the children list (cloning)
        if overwrite:
            self.children = [c for c in children]
        else:
            self.children.extend(children)

        # Avoid duplicate children, removing them
        tmp = set([])
        to_delete = []
        for i, c in enumerate(self.children):
            if c in tmp:
                to_delete.append(i)
            else:
                tmp.add(c)
        for i in reversed(to_delete):
            del self.children[i]

        # Ensure children are pointing to the parent
        for c in self.children:
            c.parent = self

    def set_parent(self, parent):
        if parent:
            parent.add_children([self])
        else:
            self.parent = None

    def set_hierarchy(self, h: Hierarchy):
        self.hierarchy = h


class Hierarchy:
    def __init__(self, name: str, root: HierarchyNode=None, htype: HierarchyType=HierarchyType.part_of):
        self.name = name
        self.root = root
        self.type = htype
        self.node_type = type(root)

    def set_root(self, root: HierarchyNode):
        self.root = root


class HierarchiesRegistry:
    def __init__(self):
        self.hierarchies = dict()
        self.node_to_hierarchies = dict() # Node object to tuple (hierarchy name, corresponding HierarchyNode)

    def add_hierarchy(self, h: Hierarchy):
        """
        Add a hierarchy to the hierarchy registry
        
        :param h: 
        :return: 
        """
        if h.name not in self.hierarchies:
            self.hierarchies[h.name] = h
        else:
            raise Exception("The hierarchy '"+h.name+' is already registered')

    def add_hierarchy_to_node(self, n: Node, h: Hierarchy, hn: HierarchyNode):
        """
        Add a hierarchy for node "n"
        
        :param n: The node 
        :param hname: The name of the hierarchy
        :param h: The node of the hierarchy in which the node is attached 
        :return: 
        """
        if n in self.node_to_hierarchies:
            hs = self.node_to_hierarchies[n]
        else:
            hs = []
            self.node_to_hierarchies[n] = hs

    def get_node_hierarchies(self, n: Node):
        pass


if __name__ == '__main__':
    """
     Build a hierarchy with 5 nodes
      * A
        * B
          * D
          * E
        * C
     Build a second hierarchy sharing two nodes
      * B
        * C
        * E
        
     Add a fund to E (it is member of only one hierarchy, no need to specify)
     Add a flow to E
     Add 
    """
    hr = HierarchiesRegistry()
    h1 = Hierarchy("H1")
    a = Node("A")
    b = Node("B")
    c = Node("C")
    d = Node("D")
    e = Node("E")
    h1a = HierarchyNode(None, "h1_a", a, h1)
    h1b = HierarchyNode(h1a, "h1_b", b)

