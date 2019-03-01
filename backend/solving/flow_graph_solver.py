"""
A solver based on the elaboration of a flow graph to quantify interfaces, connected by flow or scale relationships.
It is assumed that other kinds of relationship (part-of, upscale, ...) are translated into these two basic ones.
Another type of relationship considered is linear transform from InterfaceType to InterfaceType, which is cascaded into
appearances of its instances.

Before the elaboration of flow graphs, several preparatory steps:
* Find the separate contexts. Each context is formed by the "local", "environment" and "external" sets of processors,
  and is -totally- isolated from other contexts
  - Context defining attributes are defined in the Problem Statement command. If not, defined, a "context" attribute in
    Processors would be assumed
    If nothing is found, all Processors are assumed to be under the same context (what will happen??!!)
  - Elaborate (add necessary entities) the "environment", "external" top level processors if none have been specified.
    - Opposite processor can be specified when defining Interface
      - This attribute is taken into account if NO relationship originates or ends in this Interface. Then, a default
        relationship would be created
      - If Processors are defined for environment or
* Unexecuted model parts
  - Connection of Interfaces
  - Dataset expansion
* [Datasets]
* Scenarios
  - Parameters
* Time. Clasify QQs by time, on storage
* Observers (different versions). Take average always

"""

#import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Set, Any, Tuple, Union, Optional

from backend import case_sensitive, ureg
from backend.command_generators.parser_ast_evaluators import ast_evaluator
from backend.command_generators.parser_field_parsers import string_to_ast, expression_with_parameters, is_year, is_month
from backend.common.helper import create_dictionary, PartialRetrievalDictionary, ifnull
from backend.models.musiasem_concepts import ProblemStatement, Parameter, FactorsRelationDirectedFlowObservation, \
    FactorTypesRelationUnidirectionalLinearTransformObservation, FactorsRelationScaleObservation, Processor, \
    FactorQuantitativeObservation, Factor
from backend.model_services import get_case_study_registry_objects, State
from backend.models.musiasem_concepts_helper import find_quantitative_observations
from backend.solving.graph.computation_graph import ComputationGraph
from backend.solving.graph.flow_graph import FlowGraph


def get_circular_dependencies(parameters: Dict[str, Tuple[Any, list]]) -> list:
    # Graph, for evaluation of circular dependencies
    G = nx.DiGraph()
    for param, (_, dependencies) in parameters.items():
        for param2 in dependencies:
            G.add_edge(param2, param)  # We need "param2" to obtain "param"
    return list(nx.simple_cycles(G))


def evaluate_parameters_for_scenario(base_params: List[Parameter], scenario_params: Dict[str, str]):
    """
    Obtain a dictionary (parameter -> value), where parameter is a string and value is a literal: number, boolean,
    category or string.

    Start from the base parameters then overwrite with the values in the current scenario.

    Parameters may depend on other parameters, so this has to be considered before evaluation.
    No cycles are allowed in the dependencies, i.e., if P2 depends on P1, P1 cannot depend on P2.
    To analyze this, first expressions are evaluated, extracting which parameters appear in each of them. Then a graph
    is elaborated based on this information. Finally, an algorithm to find cycles is executed.

    :param base_params:
    :param scenario_params:
    :return:
    """
    # Create dictionary without evaluation
    result_params = create_dictionary()
    # for param in base_params:
    #     if param.default_value:
    #         result_params[param.name] = param.default_value
    result_params.update({p.name: p.default_value for p in base_params if p.default_value})

    # Overwrite with scenario expressions or constants
    result_params.update(scenario_params)

    state = State()
    known_params = create_dictionary()
    unknown_params = create_dictionary()

    # Now, evaluate ALL expressions
    for param, expression in result_params.items():
        value, ast, params, issues = evaluate_numeric_expression_with_parameters(expression, state)
        if not value:  # It is not a constant, store the parameters on which this depends
            if case_sensitive:
                unknown_params[param] = (ast, set(params))
            else:
                unknown_params[param] = (ast, set([p.lower() for p in params]))
        else:  # It is a constant, store it
            result_params[param] = value  # Overwrite
            known_params[param] = value

    cycles = get_circular_dependencies(unknown_params)
    if len(cycles) > 0:
        raise Exception(f"Parameters cannot have circular dependencies. {len(cycles)} cycles were detected: "
                        f"{':: '.join(cycles)}")

    # Initialize state with known parameters
    state.update(known_params)

    # Loop until no new parameters can be evaluated
    previous_len_unknown_params = len(unknown_params) + 1
    while len(unknown_params) < previous_len_unknown_params:
        previous_len_unknown_params = len(unknown_params)

        for param in list(unknown_params):  # A list(...) is used because the dictionary can be modified inside
            ast, params = unknown_params[param]
            if params.issubset(known_params):
                value, _, _, issues = evaluate_numeric_expression_with_parameters(ast, state)
                if not value:
                    raise Exception(f"It should be possible to evaluate the parameter '{param}'. "
                                    f"Issues: {', '.join(issues)}")
                else:
                    del unknown_params[param]
                    result_params[param] = value
                    state.set(param, value)

    if len(unknown_params) > 0:
        raise Exception(f"Could not evaluate the following parameters: {', '.join(unknown_params)}")

    return result_params


def prepare_interfaces_graph(factors: List[Factor]) -> nx.DiGraph:
    """
    Construct a DiGraph JUST with NODES, which are ALL the Factors

    :param factors:
    :return: a DiGraph
    """

    G = nx.DiGraph()
    for f in factors:
        # d = dict(interface=f)
        # G.add_node(f, **d)
        G.add_node(f)
    return G


def prepare_scales_graph(scale_relations: List[FactorsRelationScaleObservation]) -> nx.DiGraph:
    """
    This graph is used to put interfaces into scale, possibly in cascade (an Interface should not receive more than one scale).
    The scaling can then be updated for each scenario (parameters) and values can depend also on the TIME-PERIOD.
    Interfaces not involved in the graph may be removed from it.

    ADDITIONAL METHOD (set_update_scales_graph):
    * Set values, parameters, and evaluate scale expression. Initial values in the middle of a scaling chain are rejected.
      Only allowed at the beginning of a scale chain. The "scale_origin" property is set for nodes starting one of these chains.
    * Solve scales

    :param scale_relations:
    :return: nx.DiGraph (with the "scales graph")
    """
    G = nx.DiGraph()  # Start a graph

    # Auxiliary sets:
    # - "origin" only (origin-destination) nodes are sources for scale.
    # - "destination" nodes can only appear once as destination
    origin_interfaces: Set[Factor] = set()
    destination_interfaces: Set[Factor] = set()
    state = State()  # Empty State, just to be able to evaluate expressions and detect needed parameters

    for relation in scale_relations:
        value, ast, _, issues = evaluate_numeric_expression_with_parameters(relation.quantity, state)
        if value:
            ast = None

        assert(len(issues) == 0)

        # Add an edge per i2i scale relation
        G.add_edge(relation.origin, relation.destination, **dict(rel=relation, ast=ast, v=value))

        # Nodes appearing in scale are not to be removed
        # G[s.origin]["remove"] = False
        # G[s.destination]["remove"] = False

        # Nodes appearing more than once as destination is a construction error
        if relation.destination in destination_interfaces:
            raise Exception("An Interface should not appear in more than one Scale relation as destination")

        origin_interfaces.add(relation.origin)
        destination_interfaces.add(relation.destination)

    # "Beginning" nodes are marked
    nx.set_node_attributes(G, False, "beginning")
    for i in origin_interfaces.difference(destination_interfaces):
        G.nodes[i]["beginning"] = True

    # Return the resulting DiGraph plus the parameters reverse dictionary
    return G


def get_scale_beginning_interfaces(graph: nx.DiGraph):
    return set([node for node, data in graph.nodes(data=True) if data["beginning"]])


def set_update_scales_graph(graph: nx.DiGraph, params: Dict[str, Any], values: Dict[Factor, Tuple[Any, FactorQuantitativeObservation]]):
    """
    For a scaling graph:
     - set both the parameters and the values of Interfaces beginning scale-chains
     - update the scale chains accordingly

    :param graph: Graph with all scale chains
    :param params: Parameters to apply to values in edges and nodes of the graph
    :param values: Expressions (plus "unit") for beginning nodes of the graph
    :return: Nothing (the graph is updated in-place)
    """

    # Set of all nodes.
    all_interfaces = set(graph.nodes)
    # Set of "scale beginning" nodes.
    beginning_interfaces = get_scale_beginning_interfaces(graph)
    # Set of "scale following" nodes
    following_interfaces = all_interfaces.difference(beginning_interfaces)

    # Set of nodes with value
    interfaces_with_value = set(values.keys())

    # Check that all beginning interfaces have been defined (have a value)
    mandatory_to_define_all_beginning_interfaces = False

    if mandatory_to_define_all_beginning_interfaces:
        interfaces_which_should_have_a_value = beginning_interfaces.difference(interfaces_with_value)
        if interfaces_which_should_have_a_value:
            s = ", ".join([i.processor.name + ":" + i.name for i in interfaces_which_should_have_a_value])
            raise Exception("Not all scale beginning Interfaces have been assigned a value: "+s)

    # "following" interfaces in scale-chains should not have a value
    interfaces_which_should_not_have_a_value = following_interfaces.intersection(interfaces_with_value)

    if interfaces_which_should_not_have_a_value:
        s = ", ".join([i.processor.name + ":" + i.name for i in interfaces_which_should_not_have_a_value])
        raise Exception("Interfaces in scale chains cannot be assigned values: "+s)

    # Now expressions. First, prepare "state"
    state = State()
    state.update(params)

    # Evaluate (AST) all expressions from the INTERSECTION
    defined_beginning_interfaces = beginning_interfaces.intersection(interfaces_with_value)
    for i in defined_beginning_interfaces:
        expression = values[i][0]
        unit = ureg(values[i][1].attributes["unit"])
        v, _, _, issues = evaluate_numeric_expression_with_parameters(expression, state)
        if not v:
            raise Exception(f"Could not evaluate expression '{expression}': {', '.join(issues)}")
        else:
            graph.nodes[i]["v"] = v * unit

    # Evaluate all edges
    for edge, attrs in graph.edges(data=True):
        ast = attrs["ast"]
        if ast:
            v, _, _, issues = evaluate_numeric_expression_with_parameters(ast, state)
            if not v:
                raise Exception(f"Could not evaluate edge scale expression for edge ({edge[0].name}->{edge[1].name}): {', '.join(issues)}")
            else:
                edge["v"] = v

    # Now, compute values in nodes
    def compute_scaled_nodes(nodes):
        for i in nodes:
            val = graph.nodes[i]["v"]
            tmp = []
            for suc in graph.successors(i):
                # TODO Consider unit conversions, or the unit of the predecessor is inherited?
                graph.nodes[suc]["v"] *= val
                tmp.append(suc)
            compute_scaled_nodes(tmp)

    compute_scaled_nodes(defined_beginning_interfaces)


def get_observations_OLD(prd: PartialRetrievalDictionary) \
        -> Tuple[PartialRetrievalDictionary, PartialRetrievalDictionary, Dict[str, int]]:
    """
    Process All QQ observations (intensive or extensive):
    * Store in a compact way (then clear), by Time-period, by Interface, by Observer.
    * Convert to float or prepare AST
    * Store as value the result plus the QQ observation (in a tuple)

    :param prd:
    :param relative: True->a QQ observation relative to the value of another interface
    :return: another PartialRetrievalDictionary, the Observers and the Time Periods (indexed)
    """
    observations_prd = PartialRetrievalDictionary()
    relative_observations_prd = PartialRetrievalDictionary()
    time_periods: Dict[str, int] = create_dictionary()  # Dictionary of time periods and the associated IDX
    state = State()

    next_time_period_idx = 0
    for observation in find_quantitative_observations(prd, processor_instances_only=True):

        # Obtain time period index
        time = observation.attributes["time"]
        if time not in time_periods:
            time_periods[time] = next_time_period_idx
            next_time_period_idx += 1

        # Elaborate Key: Interface, Time, Observer
        key = dict(__i=observation.factor, __t=time_periods[time], __o=observation.observer)

        value, ast, _, issues = evaluate_numeric_expression_with_parameters(observation.value, state)
        if not value:
            value = ast

        # Store Key: (Value, FactorQuantitativeObservation)
        if observation.relative:
            relative_observations_prd.put(key, (value, observation))
        else:
            observations_prd.put(key, (value, observation))

    return observations_prd, relative_observations_prd, time_periods


def get_observations_by_time(prd: PartialRetrievalDictionary) -> Dict[str, List[Tuple[float, FactorQuantitativeObservation]]]:
    """
    Process All QQ observations (intensive or extensive):
    * Store in a compact way (then clear), by Time-period, by Interface, by Observer.
    * Convert to float or prepare AST
    * Store as value the result plus the QQ observation (in a tuple)

    :param prd:
    :param relative: True->a QQ observation relative to the value of another interface
    :return: another PartialRetrievalDictionary, the Observers and the Time Periods (indexed)
    """
    observations: Dict[str, List[Tuple[float, FactorQuantitativeObservation]]] = {}
    state = State()

    # Get all observations by time
    for observation in find_quantitative_observations(prd, processor_instances_only=True):

        # Try to evaluate the observation value
        value, ast, _, issues = evaluate_numeric_expression_with_parameters(observation.value, state)

        # Store: (Value, FactorQuantitativeObservation)
        time = observation.attributes["time"]
        if time not in observations:
            observations[time] = []

        observations[time].append((ifnull(value, ast), observation))

    # Check all time periods are consistent. All should be Year or Month, but not both.
    time_period_type = get_type_from_all_time_periods(list(observations.keys()))
    assert(time_period_type in ["Year", "Month"])

    # Remove generic period type and insert it into all specific periods. E.g. "Year" into "2010", "2011" and "2012"
    if time_period_type in observations:
        # Generic monthly ("Month") or annual ("Year") data
        periodic_observations = observations[time_period_type]
        del observations[time_period_type]

        for time in observations:
            observations[time] += periodic_observations

    return observations


def evaluate_numeric_expression_with_parameters(expression: Union[float, str, dict], state: State) \
        -> Tuple[Optional[float], Optional[Dict], Set, List[str]]:

    issues: List[Tuple[int, str]] = []
    ast = None
    value = None
    params = set()

    if isinstance(expression, float):
        value = expression

    elif isinstance(expression, dict):
        ast = expression
        value, params = ast_evaluator(ast, state, None, issues)
        if value:
            ast = None

    elif isinstance(expression, str):
        try:
            value = float(expression)
        except ValueError:
            ast = string_to_ast(expression_with_parameters, expression)
            value, params = ast_evaluator(ast, state, None, issues)
            if value:
                ast = None

    else:
        issues.append((3, "Invalid type for expression"))

    return value, ast, params, [i[1] for i in issues]


def get_scaled(scenarios, scenario_params, graph, observations_by_time):

    # Compute the scales for the different scenarios and time periods, and store the results in
    # another partial retrieval dictionary
    scale_beginning_interfaces = get_scale_beginning_interfaces(graph)

    scales_prd = PartialRetrievalDictionary()
    for scenario_idx, (scenario_name, scenario) in enumerate(scenarios.items()):

        for time_period, observations in observations_by_time.items():

            # Filter, and prepare dictionary for the update of the scaling
            values: Dict[Factor, Tuple[Any, FactorQuantitativeObservation]] = \
                {obs.factor: (value, obs) for value, obs in observations if obs.factor in scale_beginning_interfaces}

            # Evaluate expressions
            set_update_scales_graph(graph, scenario_params[scenario_name], values)

            # Write data back
            for n, data in graph.nodes(data=True):  # type: Factor, Dict
                # Elaborate Key: Interface, Time, Scenario
                key = dict(__i=n, __t=time_period, __s=scenario_idx)
                # Put
                scales_prd.put(key, (n, data["v"]))

    return scales_prd


def get_type_from_all_time_periods(time_periods: List[str]) -> Optional[str]:
    """ Check if all time periods are of the same period type, either Year or Month:
         - general "Year" & specific year (YYYY)
         - general "Month" & specific month (mm-YYYY or YYYY-mm, separator can be any of "-/")

    :param time_periods:
    :return:
    """
    # Based on the first element we will check the rest of elements
    period = next(iter(time_periods))

    if period == "Year" or is_year(period):
        period_type = "Year"
        period_check = is_year
    elif period == "Month" or is_month(period):
        period_type = "Month"
        period_check = is_month
    else:
        return None

    for time_period in time_periods:
        if time_period != period_type and not period_check(time_period):
            return None

    return period_type


def get_flow_graph(msm, obs_variation):
    """
    Obtain a flow graph from the MuSIASEM model
    Assign quantitative observations filtering using the observations variation parameter

    :param msm:
    :param obs_variation:
    :return:
    """

    return nx.DiGraph()


def flow_graph_solver(global_parameters: List[Parameter], problem_statement: ProblemStatement,
                      input_systems: Dict[str, Set[Processor]], state: State):
    """
    * First scales have to be solved
    * Second direct flows
    * Third conversions of flows

    Once flows have been found, Indicators have to be gathered.

    :param global_parameters: Parameters including the default value (if defined)
    :param problem_statement: ProblemStatement object, with scenarios (parameters changing the default)
                              and parameters for the solver
    :param state: State with everything
    :param input_systems: A dictionary of the different systems to be solved
    :return: Issue[]
    """

    glb_idx, _, _, _, _ = get_case_study_registry_objects(state)

    # Initialize systems
    systems: Dict[str, Dict[type, Set]] = create_dictionary()
    for i in input_systems:
        systems[i] = dict()  # A dictionary
        systems[i][Factor] = set()
        systems[i][FactorsRelationDirectedFlowObservation] = set()
        systems[i][FactorsRelationScaleObservation] = set()

    # All INTERFACE TYPE SCALE relationships
    relations_scale_it2it = glb_idx.get(FactorTypesRelationUnidirectionalLinearTransformObservation.partial_key())

    # All INTERFACE SCALE relationships (between separate processors)
    relations_scale_i2i = glb_idx.get(FactorsRelationScaleObservation.partial_key())

    for relation in relations_scale_i2i:  # type: FactorsRelationScaleObservation
        orig_system = relation.origin.processor.processor_system
        dest_system = relation.destination.processor.processor_system
        systems[orig_system][Factor].add(relation.source_factor)
        systems[dest_system][Factor].add(relation.target_factor)
        systems[orig_system][FactorsRelationScaleObservation].add(relation)
        systems[dest_system][FactorsRelationScaleObservation].add(relation)
        # TODO Expressions, AST, find parameters

    # All FLOW relationships
    relations_flow = glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key())

    for relation in relations_flow:  # type: FactorsRelationDirectedFlowObservation
        orig_system = relation.source_factor.processor.processor_system
        dest_system = relation.target_factor.processor.processor_system
        systems[orig_system][Factor].add(relation.source_factor)
        systems[orig_system][Factor].add(relation.target_factor)  # Target must be must be considered as a node in the same context!!
        systems[dest_system][Factor].add(relation.target_factor)
        systems[dest_system][Factor].add(relation.source_factor)  # Origin must be must be considered as a node in the same context!!
        systems[orig_system][FactorsRelationDirectedFlowObservation].add(relation)
        systems[dest_system][FactorsRelationDirectedFlowObservation].add(relation)
        # TODO Expressions, AST, find parameters

    # Get all interface observations separated by relative and non-relative.
    # Also resolve expressions without parameters. Cannot resolve expressions dependent from global parameters only
    # because some of them can be overridden by scenario parameters.
    # observations_prd, relative_observations_prd, time_periods = get_observations(glb_idx)
    observations_by_time = get_observations_by_time(glb_idx)

    if len(observations_by_time) == 0:
        raise Exception(f"No observations have been found. The solver cannot work.")

    # Split observations into relative and not relative
    observations_by_time_norelative = {}
    observations_by_time_relative = {}
    for time, observations in observations_by_time.items():
        observations_by_time_norelative[time] = []
        observations_by_time_relative[time] = []
        for value, obs in observations:
            if obs.relative:
                observations_by_time_relative[time].append((value, obs))
            else:
                observations_by_time_norelative[time].append((value, obs))

    # Check all time periods are consistent. All should be Year or Month, but not both.
    # time_period_type = get_type_from_all_time_periods(list(time_periods))
    # assert(time_period_type in ["Year", "Month"])

    scenario_parameters: Dict[str, Dict[str, str]] = \
        {scenario_name: evaluate_parameters_for_scenario(global_parameters, scenario_params)
         for scenario_name, scenario_params in problem_statement.scenarios.items()}

    # SCALES --------------------------

    # Obtain a i2i Scales Graph
    scale_graph = prepare_scales_graph(relations_scale_i2i)

    # periodic_observations = []
    # if time_period_type in time_periods:
    #     # Generic monthly ("Month") or annual ("Year") data
    #     periodic_observations = observations_prd.get(dict(__t=time_periods[time_period_type]))
    #     del time_periods[time_period_type]

    # Obtain the scale VALUES
    scales_prd = get_scaled(problem_statement.scenarios, scenario_parameters, scale_graph, observations_by_time_norelative)

    # FLOWS --------------------------
    for s in systems:
        # From Factors IN the context (LOCAL, ENVIRONMENT or OUTSIDE)
        # obtain a basic graph. Signal each Factor as LOCAL or EXTERNAL, and SOCIETY or ENVIRONMENT
        # basic_graph = prepare_interfaces_graph(systems[s][Factor])

        print(f"********************* SYSTEM: {s}")

        # Obtain a flow graph
        #
        # flow_graph = get_flow_graph(basic_graph, systems[s][FactorsRelationDirectedFlowObservation])
        flow_graph = FlowGraph()
        for relation in systems[s][FactorsRelationDirectedFlowObservation]:  # type: FactorsRelationDirectedFlowObservation
            # flow_graph.add_edge(relation.source_factor, relation.target_factor, weight=ifnull(relation.weight, 1.0))
            flow_graph.add_edge(relation.source_factor.full_name, relation.target_factor.full_name,
                                weight=relation.weight, reverse_weight=None)

        comp_graph, issues = flow_graph.get_computation_graph()

        for issue in issues:
            print(issue)

        print(f"****** NODES: {comp_graph.nodes}")

        # nx.draw_networkx(flow_graph)
        # plt.show()

        # TODO Expand flow graph with it2it transforms

        # Split flow graphs
        for scenario_idx, (scenario_name, scenario) in enumerate(problem_statement.scenarios.items()):

            print(f"********************* SCENARIO: {scenario_name}")

            scenario_state = State()
            scenario_state.update(scenario_parameters[scenario_name])

            for time_period, observations in observations_by_time_norelative.items():

                print(f"********************* TIME PERIOD: {time_period}")

                # Obtain quantities, separate by time.
                # Also, identify SCALES i2i, intraprocessor (from Observations)
                scales = {fact: val for fact, val in scales_prd.get(dict(__t=time_period, __s=scenario_idx))}

                # Final values are taken from "scales" or from "observations" that need to computed
                graph_params = {}
                for expression, obs in observations:
                    if obs.factor.full_name not in comp_graph.nodes:
                        print(f"WARNING: observation '{obs.factor.full_name}' is not taken into account.")
                    else:
                        if scales.get(obs.factor):
                            graph_params[obs.factor.full_name] = scales[obs.factor]
                        else:
                            value, ast, _, issues = evaluate_numeric_expression_with_parameters(expression, scenario_state)
                            if not value:
                                raise Exception(f"Cannot evaluate expression '{expression}'")

                            graph_params[obs.factor.full_name] = value

                # ----------------------------------------------------

                compute_nodes = [n for n in comp_graph.nodes if not graph_params.get(n)]

                # Compute the missing information with the computation graph
                if len(compute_nodes) > 0:

                    print(f"****** UNKNOWN NODES: {compute_nodes}")
                    print(f"****** PARAMS: {graph_params.keys()}")

                    conflicts = comp_graph.compute_param_conflicts(set(graph_params.keys()))

                    for i, (param, values) in enumerate(conflicts.items()):
                        print(f"Conflict {i + 1}: {param} -> {values}")

                    combinations = ComputationGraph.compute_param_combinations(conflicts)

                    for i, combination in enumerate(combinations):
                        print(f"Combination {i}: {combination}")

                        filtered_params = {k: v for k, v in graph_params.items() if k in combination}
                        results, values = comp_graph.compute_values(compute_nodes, filtered_params)

                        print(f'  results={results}')

                # TODO Overwrite "obs" with "scales" results
                # TODO Put observations into the flow-graph


                # TODO Put processors into scale (intensive to extensive conversion)
                # scale_unit_processors(flow_graph, params, relative_observations_prd)

                # for sub_fg in nx.weakly_connected_component_subgraphs(flow_graph):
                    # TODO Elaborate information flow graph
                    #      Cycles allowed?
                    # ifg = get_information_flow_graph(sub_fg)
                    # TODO Solve information flow graph. From all possible combinations:
                    #  bottom-up if top-down USE
                    #  bottom-up if top-down DO NOT USE
                    #  top-down  if bottom-up USE
                    #  top-down  if bottom-up DO NOT USE
                    # solve_flow_graph(sub_fg, ifg)  # Each value: Interface, Scenario, Time, Given/Computed -> VALUE (or UNDEFINED)
                    # TODO Put results back


        # TODO INDICATORS --- (INSIDE FLOWS)

    return []
