from typing import List, Union
from abc import ABCMeta, abstractmethod
import networkx as nx

from backend.common.helper import create_dictionary
from backend.model.memory.musiasem_concepts import Processor, Observer, FactorType, Factor, \
    FactorQuantitativeObservation, FactorTypesRelationUnidirectionalLinearTransformObservation, \
    ProcessorsRelationPartOfObservation, ProcessorsRelationUndirectedFlowObservation, \
    ProcessorsRelationUpscaleObservation, FactorsRelationDirectedFlowObservation
from backend.model.memory.musiasem_concepts_helper import find_or_create_observable
from backend.model_services import get_case_study_registry_objects
from backend.model_services.workspace import State
from backend.restful_service.serialization import deserialize_state

"""

This module will contain functions to compose graphs for different purposes

* For Internal/External Processes capable of exploiting the graph topology and associated attributes

* For deduction of Factors:
  - Nodes: Factors
  - Edges: flow relations, relations between FactorTypes (hierarchy, expressions in hierarchies, linear transforms)


* For summary visualization of flow relation between Processors 
  - Nodes: Processors
  - Edges: Flows between factors, several flows between the same pair of processor are summarized into one

* Visualization of graph with hierarchy of processors 
  - Nodes: Processors
  - Edges: part-of relations
* Visualization of Processors and their factors
  - Nodes: Processors and Factors
  - Edges: Processors to Factors, flows Factor to Factor

Allow filtering:
  - By Observer
  - By other criteria 


>> Si hay expresiones no lineales, np hay una manera directa de expresar en el grafo <<

Mapa

Parámetros - Variables SymPy
Factores - Variables SymPy

ECUACIONES DIRECTAMENTE

Factores -> Variables
Parámetros -> Inputs

Observaciones extensivas -> Valores para las variables
Observaciones intensivas -> ECUACIÓN: F2 = (V)*F1
Flujo entre factores: padre <- hijos: F = sum(Fhijos)
Flujo entre factores: SPLIT: F1 = W1*F, F2 = W2*F ó F = F1/W1 + F2/W2 + ...
Flujo entre factores: JOIN : F  = W1*F1 + W2*F2 + ...
Ambos son redundantes

F1  G1
F2  G2
G1 = W1*F1 + W2*F2
G2 = (1-W1)*F1 + (1-W2)*F2  
"""


class IQueryObjects(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, object_classes: List, filt: Union[dict, str]) -> str:
        """
        Query state to obtain objects of types enumerated in "object_classes", applying a filter
        In general, interface to pass state, select criteria (which kinds of objects to retrieve) and
        filter criteria (which of these objects to obtain).

        :param object_classes: A list with the names/codes of the types of objects to obtain
        :param filt: A way of expressing a filter of objects of each of the classes to be retrieved
        :return:
        """
        pass


class BasicQuery(IQueryObjects):
    def __init__(self, state: State):
        self._state = state
        self._registry, self._p_sets, self._hierarchies, self._datasets, self._mappings = get_case_study_registry_objects(state)

    def execute(self, object_classes: List, filt: Union[dict, str]) -> str:
        requested = {}
        types = [Observer, Processor, FactorType, Factor,
                 FactorQuantitativeObservation, FactorTypesRelationUnidirectionalLinearTransformObservation,
                 ProcessorsRelationPartOfObservation, ProcessorsRelationUpscaleObservation,
                 ProcessorsRelationUndirectedFlowObservation,
                 FactorsRelationDirectedFlowObservation]
        for o_class in object_classes:
            for t in types:
                if (isinstance(o_class, str) and o_class.lower() == t.__name__.lower()) or \
                        (isinstance(o_class, type) and o_class == t):
                    requested[t] = None

        if Observer in requested:
            # Obtain All Observers
            oers = self._registry.get(Observer.partial_key())
            # Apply filter
            if "observer_name" in filt:
                oers = [o for o in oers if o.name.lower() == filt["observer_name"]]
            # Store result
            requested[Observer] = oers
        if Processor in requested:
            # Obtain All Processors
            procs = set(self._registry.get(Processor.partial_key()))
            # TODO Apply filter
            # Store result
            requested[Processor] = procs
        if FactorType in requested:
            # Obtain FactorTypes
            fts = set(self._registry.get(FactorType.partial_key()))
            # TODO Apply filter
            # Store result
            requested[FactorType] = fts
        if Factor in requested:
            # Obtain Factors
            fs = self._registry.get(Factor.partial_key())
            # TODO Apply filter
            # Store result
            requested[Factor] = fs
        if FactorQuantitativeObservation in requested:
            # Obtain Observations
            qqs = self._registry.get(FactorQuantitativeObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[FactorQuantitativeObservation] = qqs
        if FactorTypesRelationUnidirectionalLinearTransformObservation in requested:
            # Obtain Observations
            ftlts = self._registry.get(FactorTypesRelationUnidirectionalLinearTransformObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[FactorTypesRelationUnidirectionalLinearTransformObservation] = ftlts
        if ProcessorsRelationPartOfObservation in requested:
            # Obtain Observations
            pos = self._registry.get(ProcessorsRelationPartOfObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[ProcessorsRelationPartOfObservation] = pos
        if ProcessorsRelationUndirectedFlowObservation in requested:
            # Obtain Observations
            ufs = self._registry.get(ProcessorsRelationUndirectedFlowObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[ProcessorsRelationUndirectedFlowObservation] = ufs
        if ProcessorsRelationUpscaleObservation in requested:
            # Obtain Observations
            upss = self._registry.get(ProcessorsRelationUpscaleObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[ProcessorsRelationUpscaleObservation] = upss
        if FactorsRelationDirectedFlowObservation in requested:
            # Obtain Observations
            dfs = self._registry.get(FactorsRelationDirectedFlowObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[FactorsRelationDirectedFlowObservation] = dfs

        return requested


def get_processor_id(p: Processor):
    return p.name.lower()


def get_factor_id(f_: Union[Factor, Processor], ft: FactorType=None):
    if isinstance(f_, Factor):
        return (f_.processor.name + ":" + f_.taxon.name).lower()
    elif isinstance(f_, Processor) and isinstance(ft, FactorType):
        return (f_.name + ":" + ft.name).lower()


def get_factor_type_id(ft: (FactorType, Factor)):
    if isinstance(ft, FactorType):
        return ":"+ft.name.lower()
    elif isinstance(ft, Factor):
        return ":" + ft.taxon.name.lower()


def processor_to_dict(p: Processor):
    return dict(name=get_processor_id(p), ident=p.ident)


def factor_to_dict(f_: Factor):
    return dict(name=get_factor_id(f_), rep=str(f_), ident=f_.ident)


def construct_solve_graph(state: State, query: IQueryObjects, filt: Union[str, dict]):
    """
    Prepare a graph from which conclusions about factors can be extracted

    :param state: State
    :param query: A IQueryObjects instance (which has been already injected the state)
    :param filt: A filter to be passed to the query instance
    :return:
    """
    include_processors = False  # For clarity, include processors nodes, as a way to visualize grouped factors
    will_write = True  # For debugging purposes, affects how the properties attached to nodes and edges are elaborated
    # Format
    stated_factor_no_observation = dict(graphics={'fill': "#999900"})  # Golden
    stated_factor_some_observation = dict(graphics={'fill': "#ffff00"})  # Yellow
    qq_attached_to_factor = dict(graphics={'fill': "#eeee00", "type": "ellipse"})  # Less bright Yellow
    non_stated_factor = dict(graphics={'fill': "#999999"})
    a_processor = dict(graphics={"type": "hexagon", "color": "#aa2211"})

    edge_from_factor_type = dict(graphics={"fill": "#ff0000", "width": 1, "targetArrow": "standard"})
    edge_processor_to_factor = dict(graphics={"fill": "#ff00ff", "width": 3, "targetArrow": "standard"})
    edge_factors_flow = dict(graphics={"fill": "#000000", "width": 5, "targetArrow": "standard"})
    edge_factors_upscale = dict(graphics={"fill": "#333333", "width": 3, "targetArrow": "standard"})
    edge_factors_relative_to = dict(graphics={"fill": "#00ffff", "width": 3, "targetArrow": "standard"})
    edge_factor_value = dict(graphics={"fill": "#aaaaaa", "width": 1, "targetArrow": "standard"})

    glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
    # Obtain the information needed to elaborate the graph
    objs = query.execute([Processor, Factor, FactorType,
                          FactorQuantitativeObservation,
                          FactorTypesRelationUnidirectionalLinearTransformObservation,
                          ProcessorsRelationPartOfObservation, ProcessorsRelationUpscaleObservation,
                          ProcessorsRelationUndirectedFlowObservation,
                          FactorsRelationDirectedFlowObservation
                          ],
                         filt
                         )

    # Index quantitative observations.
    # Also, mark Factors having QQs (later this will serve to color differently these nodes)
    qqs = {}
    qq_cont = 0
    factors_with_some_observation = set()
    for o in objs[FactorQuantitativeObservation]:
        # Index quantitative observations.
        if "relative_to" in o.attributes and o.attributes["relative_to"]:
            continue  # Do not index intensive quantities, because they are translated as edges in the graph
        if o.factor in qqs:
            lst = qqs[o.factor]
        else:
            lst = []
            qqs[o.factor] = lst
        lst.append(o)
        # Mark Factors having QQs (later this will serve to color differently these nodes)
        factors_with_some_observation.add(o.factor)

    # ---- MAIN GRAPH: Factors and relations between them ----
    the_node_names_set = set()
    # --   Nodes: "Factor"s passing the filter, and QQs associated to some of the Factors
    n = []
    e = []
    f_types = {}  # Contains a list of Factors for each FactorType
    p_factors = {}  # Contains a list of Factors per Processor
    factors = create_dictionary() # Factor_ID -> Factor
    for f in objs[Factor]:
        f_id = get_factor_id(f)
        factors[f_id] = f  # Dictionary Factor_ID -> Factor
        # f_types
        if f.taxon in f_types:
            lst = f_types[f.taxon]
        else:
            lst = []
            f_types[f.taxon] = lst
        lst.append(f)
        # p_factors
        if f.processor in p_factors:
            lst = p_factors[f.processor]
        else:
            lst = []
            p_factors[f.processor] = lst
        lst.append(f)

        # Node
        the_node_names_set.add(f_id)
        if will_write:
            n.append((f_id, stated_factor_some_observation if f in factors_with_some_observation else stated_factor_no_observation))
            if f in qqs:
                for qq in qqs[f]:
                    value = str(qq_cont) + ": " + str(qq.value)
                    n.append((value, qq_attached_to_factor))
                    e.append((value, f_id, {"w": "", "label": "", **edge_factor_value}))
                    qq_cont += 1
        else:
            d = dict(factor=factor_to_dict(f), observations=qqs[f_id] if f_id in qqs else [])
            n.append((f_id, d))

    # --   Edges
    # "Relative to" relation (internal to the Processor) -> Intensive to Extensive
    for o in objs[FactorQuantitativeObservation]:
        if "relative_to" in o.attributes and o.attributes["relative_to"]:
            defining_factor = o.attributes["relative_to"]  # TODO Parse "defining_factor", it can be composed of the factor name AND the unit
            f_id = get_factor_id(o.factor)
            # Check that both "defining_factor" and "f_id" exist in the nodes list (using "factors")
            factors[defining_factor]
            factors[f_id]
            weight = "1"
            e.append((defining_factor, f_id, {"w": weight, "label": weight, **edge_factors_relative_to}))

    # Directed Flows between Factors
    for df in objs[FactorsRelationDirectedFlowObservation]:
        sf = get_factor_id(df.source_factor)
        tf = get_factor_id(df.target_factor)
        # Check that both "sf" and "tf" exist in the nodes list (using "factors")
        factors[sf]
        factors[tf]
        weight = df.weight if df.weight else "1"
        e.append((sf, tf, {"w": weight, "label": weight, **edge_factors_flow}))

    # TODO Consider Upscale relations
    # e.append((..., ..., {"w": upscale_weight, "label": upscale_weight, **edge_factors_upscale}))

    # -- Create the graph
    factors_graph = nx.DiGraph()
    factors_graph.add_nodes_from(n)
    factors_graph.add_edges_from(e)

    # nx.write_gml(factors_graph, "/home/rnebot/IntermediateGraph.gml")

    # ---- AUXILIARY GRAPH: FACTOR TYPES AND THEIR INTERRELATIONS ----
    n = []
    e = []
    # --   Nodes: "FactorType"s passing the filter
    for ft in objs[FactorType]:
        n.append((get_factor_type_id(ft), dict(factor_type=ft)))

    # --   Edges
    # Hierarchy and expressions stated in the hierarchy
    ft_in = {}  # Because FactorTypes cannot be both in hierarchy AND expression, marks if it has been specified one was, to raise an error if it is specified also the other way
    for ft in objs[FactorType]:
        ft_id = get_factor_type_id(ft)
        if ft.expression:
            if ft not in ft_in:
                # TODO Create one or more relations, from other FactorTypes (same Hierarchy) to this one
                # TODO The expression can only be a sum of FactorTypes (same Hierarchy)
                ft_in[ft] = "expression"
                # TODO Check that both "ft-id" and "..." exist in the nodes list (keep a temporary set)
                # weight = ...
                # e.append((ft_id, ..., {"w": weight, "label": weight, "origin": ft, "destination": ...}))

        if ft.parent:
            if ft.parent not in ft_in or (ft.parent in ft_in and ft_in[ft.parent] == "hierarchy"):
                # Create an edge from this FactorType
                ft_in[ft.parent] = "hierarchy"
                parent_ft_id = get_factor_type_id(ft.parent)
                # TODO Check that both "ft-id" and "parent_ft_id" exist in the nodes list (keep a temporary set)
                # Add the edge
                e.append((ft_id, parent_ft_id, {"w": "1", "origin": ft, "destination": ft.parent}))
            else:
                raise Exception("The FactorType '"+ft_id+"' has been specified by an expression, it cannot be parent.")
    # Linear transformations
    for f_rel in objs[FactorTypesRelationUnidirectionalLinearTransformObservation]:
        origin = get_factor_type_id(f_rel.origin)
        destination = get_factor_type_id(f_rel.destination)
        e.append((origin, destination, {"w": f_rel.weight, "label": f_rel.weight, "origin": f_rel.origin, "destination": f_rel.destination}))

    # --   Create FACTOR TYPES graph
    factor_types_graph = nx.DiGraph()
    factor_types_graph.add_nodes_from(n)
    factor_types_graph.add_edges_from(e)

    # ---- Obtain weakly connected components of factor_types_graph ----
    factor_types_subgraphs = list(nx.weakly_connected_component_subgraphs(factor_types_graph))

    # ---- EXPAND FACTORS GRAPH with FACTOR TYPES RELATIONS ----
    # The idea is: clone a FactorTypes subgraph if a Factor instances some of its member nodes
    # This cloning process can imply creating NEW Factors

    the_new_node_names_set = set()

    # Obtain weak components of the main graph. Each can be considered separately

    # for sg in nx.weakly_connected_component_subgraphs(factors_graph):  # For each subgraph
    #     print("--------------------------------")
    #     for n in sg.nodes():
    #         print(n)

    sg_list = []  # List of modified (augmented) subgraphs
    for sg in nx.weakly_connected_component_subgraphs(factors_graph):  # For each subgraph
        sg_list.append(sg)
        # Consider each Factor of the subgraph
        unprocessed_factors = set(sg.nodes())
        while unprocessed_factors:  # For each UNPROCESSED Factor
            tmp = unprocessed_factors.pop()  # Get next unprocessed "factor name"
            if tmp not in factors:  # QQ Observations are in the graph and not in "factors". The same with Processors
                continue
            f_ = factors[tmp]  # Obtain Factor from "factor name"
            ft_id = get_factor_type_id(f_)  # Obtain FactorType name from Factor
            # Iterate through FactorTypes and check if the Factor appears
            for sg2 in factor_types_subgraphs:  # Each FactorTypes subgraph
                if ft_id in sg2:  # If the current Factor is in the subgraph
                    if len(sg2.nodes()) > 1:  # If the FactorType subgraph has at least two nodes
                        # CLONE FACTOR TYPES SUBGRAPH
                        # Nodes. Create if not present already
                        n = []
                        e = []
                        for n2, attrs in sg2.nodes().items():  # Each node in the FactorTypes subgraph
                            ft_ = attrs["factor_type"]
                            f_id = get_factor_id(f_.processor, ft_)
                            if f_id not in sg:  # If the FactorType is not
                                # Create Factor, from processor and ft_ -> f_new
                                _, _, f_new = find_or_create_observable(state, name=f_id, source="solver")
                                factors[f_id] = f_new
                                if f_id not in the_node_names_set:
                                    if will_write:
                                        n.append((f_id, non_stated_factor))
                                    else:
                                        d = dict(factor=factor_to_dict(f_new), observations=[])
                                        n.append((f_id, d))
                                if f_id not in the_node_names_set:
                                    the_new_node_names_set.add(f_id)
                                the_node_names_set.add(f_id)
                            else:
                                unprocessed_factors.discard(f_id)
                        # Edges. Create relations between factors
                        for r2, w_ in sg2.edges().items():
                            # Find origin and destination nodes. Copy weight. Adapt weight? If it refers to a FactorType, instance it?
                            origin = get_factor_id(f_.processor, w_["origin"])
                            destination = get_factor_id(f_.processor, w_["destination"])
                            if origin in the_new_node_names_set or destination in the_new_node_names_set:
                                graphics = edge_from_factor_type
                            else:
                                graphics = {}
                            e.append((origin, destination, {"w": w_["w"], "label": w_["w"], **graphics}))
                        sg.add_nodes_from(n)
                        sg.add_edges_from(e)
                        break

    # for sg in sg_list:
    #     print("--------------------------------")
    #     for n in sg.nodes():
    #         print(n)

    # Recompose the original graph
    factors_graph = nx.compose_all(sg_list)

    # ----
    # Add "Processor"s just as a way to visualize grouping of factors (they do not influence relations between factors)
    # -
    if include_processors:
        n = []
        e = []
        for p in objs[Processor]:
            p_id = get_processor_id(p)
            if will_write:
                n.append((p_id, a_processor))
            else:
                n.append((p_id, processor_to_dict(p)))
            # Edges between Processors and Factors
            for f in p_factors[p]:
                f_id = get_factor_id(f)
                e.append((p_id, f_id, edge_processor_to_factor))
        factors_graph.add_nodes_from(n)
        factors_graph.add_edges_from(e)

    #
    # for ft in objs[FactorType]:
    #     if ft.parent:
    #         # Check which Factors are instances of this FactorType
    #         if ft in f_types:
    #             for f in f_types[ft]:
    #                 # Check if the processor contains the parent Factor
    #                 processor_factors = p_factors[f.processor]
    #                 if ft.parent not in processor_factors:
    #                     factor_data = (f.processor, ft)
    #                 else:
    #                     factor_data = None
    #                 create_factor = f in qqs  # If there is some Observation
    #                 create_factor = True # Force creation
    #
    #
    #         # Consider the creation of a relation
    #         # Consider also the creation of a new Factor (a new Node for now): if the child has some observation for sure (maybe a child of the child had an observation, so it is the same)
    #         ft_id =
    #     ft_id =

    # # Plot graph to file
    # import matplotlib.pyplot as plt
    # ax = plt.subplot(111)
    # ax.set_title('Soslaires Graph', fontsize=10)
    # nx.draw(factors_graph, with_labels=True)
    # plt.savefig("/home/rnebot/Graph.png", format="PNG")

    nx.write_gml(factors_graph, "/home/rnebot/Graph.gml")

    # Legend graph
    n = []
    e = []
    n.append(("Factor with Observation", stated_factor_some_observation))
    n.append(("Factor with No Observation", stated_factor_no_observation))
    if include_processors:
        n.append(("Processor", a_processor))
    n.append(("Factor from FactorType", non_stated_factor))
    n.append(("QQ Observation", qq_attached_to_factor))
    n.append(("QQ Intensive Observation", qq_attached_to_factor))

    e.append(("A Factor", "Another Factor", {"label": "Flow between Factors, attaching the weight", **edge_factors_flow}))
    e.append(("Factor #1", "Factor #2", {"label": "Relation from a FactorType", **edge_from_factor_type}))
    if include_processors:
        e.append(("Processor", "A Factor", {"label": "Link from Processor to Factor", **edge_processor_to_factor}))
    e.append(("A Factor", "Same Factor in another processor", {"label": "Upscale a Factor in two processors", **edge_factors_upscale}))
    e.append(("Factor with Observation", "QQ Intensive Observation", {"label": "Observation proportional to extensive value of factor same processor", **edge_factors_relative_to}))
    e.append(("QQ Observation", "A Factor", {"label": "A QQ Observation", **edge_factor_value}))
    factors_graph = nx.DiGraph()
    factors_graph.add_nodes_from(n)
    factors_graph.add_edges_from(e)
    nx.write_gml(factors_graph, "/home/rnebot/LegendGraph.gml")


# Deserialize previously recorded Soslaires State
with open("/home/rnebot/GoogleDrive/AA_MAGIC/Soslaires.serialized", "r") as file:
    s = file.read()
state = deserialize_state(s)
# Create a Query and execute a query
query = BasicQuery(state)
construct_solve_graph(state, query, None)
