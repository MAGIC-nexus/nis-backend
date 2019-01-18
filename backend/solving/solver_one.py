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

import networkx as nx
from typing import Dict, List, Set, Any, Tuple

from backend import case_sensitive, ureg
from backend.command_generators.parser_ast_evaluators import ast_evaluator
from backend.command_generators.parser_field_parsers import string_to_ast, expression_with_parameters
from backend.common.helper import strcmp, create_dictionary, PartialRetrievalDictionary
from backend.models.musiasem_concepts import ProblemStatement, Parameter, FactorsRelationDirectedFlowObservation, \
    FactorTypesRelationUnidirectionalLinearTransformObservation, FactorsRelationScaleObservation, Processor, \
    FactorQuantitativeObservation, Factor
from backend.model_services import get_case_study_registry_objects, State


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
    for param in base_params:
        if param.default_value:
            result_params[param.name] = param.default_value

    # Overwrite with scenario expressions or constants
    for param_name in scenario_params:
        result_params[param_name] = scenario_params[param_name]

    state = State()
    # Now, evaluate ALL expressions
    parameters = create_dictionary()  # From parameter to parameters appearing in the expression, plus the AST
    for param, expression in result_params.items():
        # AST from expression
        ast = string_to_ast(expression_with_parameters, expression)
        # Evaluate AST
        issues = []
        v, params = ast_evaluator(ast, state, None, issues)
        if not v:  # It is not a constant, store the parameters on which this depends
            if case_sensitive:
                parameters[param] = (ast, set(params))
            else:
                parameters[param] = (ast, set([p.lower() for p in params]))
        else:  # It is a constant, store it
            parameters[param] = v
            result_params[param] = v  # Overwrite

    # Graph, for evaluation of circular dependencies
    G = nx.DiGraph()
    for param, dependencies in parameters.items():
        for param2 in dependencies:
            G.add_edge(param2, param)  # We need "param2" to obtain "param"
    cycles = list(nx.simple_cycles(G))
    if len(cycles) > 0:
        raise Exception("Parameters cannot have circular dependencies. "+str(len(cycles))+" cycles were detected: "+
                        ":: ".join(cycles))

    # Initialize state and other auxiliary variables for the evaluation
    known_params = set()
    unknown_params = create_dictionary()
    for param, value in parameters.items():
        if not isinstance(value, tuple):
            state.set(param, value)
            if case_sensitive:
                known_params.add(param)
            else:
                known_params.add(param.lower())
        else:
            unknown_params[param] = value

    # Loop until no new parameters can be evaluated
    change = True
    while change:
        change = False
        for param in list(unknown_params):  # A list(...) is used because the dictionary can be modified inside
            value = unknown_params[param]
            if value[1].issubset(known_params):
                issues = []
                v, params = ast_evaluator(value[0], state, None, issues)
                if not v:
                    raise Exception("It should be possible to evaluate the parameter '"+param+"'. Issues: "+", ".join(issues))
                else:
                    state.set(param, v)
                    del unknown_params[param]
                    result_params[param] = v
                    change = True

    if len(unknown_params) > 0:
        raise Exception("Could not evaluate the following parameters: "+", ".join(unknown_params))

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
    origin_interfaces = set()
    destination_interfaces = set()
    state = State()  # Empty State, just to be able to evaluate expressions and detect needed parameters
    for s in scale_relations:
        try:  # Try the most frequent expression, a simple number. If not, try the rest
            ast = None
            v = float(s.quantity)
        except:
            # AST from quantity
            ast = string_to_ast(expression_with_parameters, s.quantity)
            issues = []
            v, params = ast_evaluator(ast, state, None, issues)
            if v:
                ast = None

        # Add an edge per i2i scale relation
        G.add_edge(s.origin, s.destination, dict(rel=s, ast=ast, v=v))

        # Nodes appearing in scale are not to be removed
        G[s.origin]["remove"] = False
        G[s.destination]["remove"] = False
        # Nodes appearing more than once as destination is a construction error
        if s.destination in destination_interfaces:
            raise Exception("An Interface should not appear in more than one Scale relation as destination")

        origin_interfaces.add(s.origin)
        destination_interfaces.add(s.destination)

    # "Beginning" nodes are marked
    nx.set_node_attributes(G, "B", False)
    for i in origin_interfaces.difference(destination_interfaces):
        G.nodes[i]["B"] = True

    # Return the resulting DiGraph plus the parameters reverse dictionary
    return G


def get_scale_beginning_interaces(sc: nx.DiGraph):
    return set([x for x, y in sc.nodes(data=True) if y["B"]])


def set_update_scales_graph(sc: nx.DiGraph, params: Dict[str, Any], values: Dict[Factor, Tuple[Any, FactorQuantitativeObservation]]):
    """
    For a scaling graph:
     - set both the parameters and the values of Interfaces beginning scale-chains
     - update the scale chains accordingly

    :param sc: Graph with all scale chains
    :param params: Parameters to apply to values in edges and nodes of the graph
    :param values: Expressions (plus "unit") for beginning nodes of the graph
    :return: Nothing (the graph is updated in-place)
    """

    # Set of nodes. Set of "scale beginning" nodes. Set of "scale following" nodes
    scale_interfaces = set(sc.nodes)
    scale_beginning_interfaces = get_scale_beginning_interaces(sc)
    scale_following_interfaces = scale_interfaces.difference(scale_beginning_interfaces)

    # Set of nodes with value
    interfaces_with_value = set(values.keys())

    # Check that all beginning interfaces have been defined (have a value)
    mandatory_to_define_all_beginning_interfaces = False
    if mandatory_to_define_all_beginning_interfaces:
        if not scale_beginning_interfaces.issubset(interfaces_with_value):
            s = ", ".join([i.processor.name + ":" + i.name for i in scale_beginning_interfaces.difference(interfaces_with_value)])
            raise Exception("Not all scale beginning Interfaces have been assigned a value: "+s)

    # "following" interfaces in scale-chains should not have a value
    interfaces_which_should_not_have_a_value = scale_following_interfaces.intersection(interfaces_with_value)
    if interfaces_which_should_not_have_a_value:
        s = ", ".join([i.processor.name + ":" + i.name for i in interfaces_which_should_not_have_a_value])
        raise Exception("Interfaces in scale chains cannot be assigned values: "+s)

    # Now expressions. First, prepare "state"
    state = State()
    for param, value in params.items():
        state.set(param, value)

    # Evaluate (AST) all expressions from the INTERSECTION
    defined_interfaces = scale_beginning_interfaces.intersection(interfaces_with_value)
    for i in defined_interfaces:
        expression = values[i][0]
        unit = ureg(values[i][1]._unit)
        if isinstance(expression, float):  # Literal
            sc.nodes[i]["v"] = expression * unit
        elif isinstance(expression, dict):  # AST
            issues = []
            v, params = ast_evaluator(ast, state, None, issues)
            if not v:
                raise Exception("Could not evaluate expression '" + expression + "': " + ", ".join(issues))
            else:
                sc.nodes[i]["v"] = v * unit
        elif isinstance(expression, str):
            try:  # Try the most frequent expression, a simple number. If not, try the rest
                sc.nodes[i]["v"] = float(expression)
            except:
                # AST from expression
                ast = string_to_ast(expression_with_parameters, expression)
                # Evaluate AST
                issues = []
                v, params = ast_evaluator(ast, state, None, issues)
                if not v:
                    raise Exception("Could not evaluate expression '"+expression+"': "+", ".join(issues))
                else:
                    sc.nodes[i]["v"] = v * unit

    # Evaluate all edges
    for e, attrs in sc.edges(data=True):
        ast = attrs["ast"]
        if ast:
            # Evaluate AST
            issues = []
            v, params = ast_evaluator(ast, state, None, issues)
            if not v:
                raise Exception("Could not evaluate edge scale expression for edge ("+e[0].name+"->"+e[1].name+": "+", ".join(issues))
            else:
                sc.edges[i]["v"] = v

    # Now, compute values in nodes
    def compute_scaled_nodes(nodes):
        for i in nodes:
            x = sc.nodes[i]["v"]
            tmp = []
            for oe, attrs in sc.out_edges(i, data=True):
                # TODO Consider unit conversions, or the unit of the predecesor is inherited?
                sc.nodes[oe[1]]["v"] = x * attrs["v"]
                tmp.append(oe[1])
            compute_scaled_nodes(tmp)

    compute_scaled_nodes(defined_interfaces)


def get_observations(prd0: PartialRetrievalDictionary, relative=False):
    """
    Process All QQ observations (intensive or extensive):
    * Store in a compact way (then clear), by Time-period, by Interface, by Observer.
    * Convert to float or prepare AST
    * Store as value the result plus the QQ observation (in a tuple)

    :param prd0:
    :param relative: True->a QQ observation relative to the value of another interface
    :return: another PartialRetrievalDictionary, the Observers and the Time Periods (indexed)
    """

    observations_prd = PartialRetrievalDictionary()

    state = State()
    oers = set()  # Set of Observers
    time_periods = create_dictionary()  # Dictionary of time periods and the associated IDX
    time_periods_idx = 0
    for o in prd0.get(FactorQuantitativeObservation.partial_key(relative=relative)):
        # Skip observations of Interfaces in Processors OUT of accounting
        if o.factor.processor.instance_or_archetype == "Archetype":
            continue

        # Store Observer
        oers.add(o.observer)

        # Obtain time period index
        if o._time in time_periods:
            time_idx = time_periods[o._time]
        else:
            time_periods[o._time] = time_periods_idx
            time_periods_idx += 1

        # Elaborate Key: Interface, Time, Observer
        key = dict(__i=o.factor, __t=time_idx, __o=o.observer)

        # Elaborate Value
        try:
            value = float(o.value)
        except:
            # AST from quantity
            ast = string_to_ast(expression_with_parameters, s.quantity)
            issues = []
            value, params = ast_evaluator(ast, state, None, issues)
            if not value:
                value = ast

        # Store Key: (Value, FactorQuantitativeObservation)
        observations_prd.put(key, (value, o))

    return observations_prd, oers, time_periods


def get_scaled(global_parameters, scenarios, sg, observations_prd, time_periods):

    # Generic month data
    month_data = observations_prd.get(dict(__t=time_periods["Month"]))
    # Generic year data
    year_data = observations_prd.get(dict(__t=time_periods["Year"]))

    # Compute the scales for the different scenarios and time periods, and store the results in
    # another partial retrieval dictionary
    scale_beginning_interfaces = get_scale_beginning_interaces(sg)
    scales_prd = PartialRetrievalDictionary()
    for scenario_idx, (scenario_name, scenario) in enumerate(scenarios.items()):
        params = evaluate_parameters_for_scenario(global_parameters, scenario)
        for time_period in sorted(time_periods):
            # Obtain data
            time_idx = time_periods[time_period]
            obs = observations_prd.get(dict(__t=time_idx))  # type: List[Tuple[Any, FactorQuantitativeObservation]]
            if is_year(time_period):
                obs.append(year_data)
            elif is_month(time_period):
                obs.append(month_data)

            # Filter, and prepare dictionary for the update of the scaling
            values = {o[1].factor: (o[0], o[1]._unit) for o in obs if o[1].factor in scale_beginning_interfaces}

            # Evaluate expressions
            set_update_scales_graph(sg, params, values)

            # Write data back
            for n, data in sg.nodes(data=True):
                # Elaborate Key: Interface, Time, Scenario
                key = dict(__i=n, __t=time_idx, __s=scenario_idx)
                # Value
                value = data["v"]
                # Put
                scales_prd.put(key, value)

    return scales_prd


def flow_graph_solver(global_parameters: List[Parameter], problem_statement: ProblemStatement,
                      in_systems: Dict[str, Set[Processor]], state: State):
    """
    * First scales have to be solved
    * Second direct flows
    * Third conversions of flows

    Once flows have been found, Indicators have to be gathered.

    :param global_parameters: Parameters including the default value (if defined)
    :param problem_statement: ProblemStatement object, with scenarios (parameters changing the default)
                              and parameters for the solver
    :param state: State with everything
    :param in_systems: A dictionary of the different systems to be solved
    :return: Issue[]
    """

    glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)

    systems = create_dictionary()
    for c in in_systems:
        systems[c] = dict()  # A dictionary
        systems[c][Factor] = set()
        systems[c][FactorsRelationDirectedFlowObservation] = set()
        systems[c][FactorsRelationScaleObservation] = set()

    # All INTERFACE TYPE SCALE relationships
    it2it = glb_idx.get(FactorTypesRelationUnidirectionalLinearTransformObservation.partial_key())

    # All INTERFACE SCALE relationships (between separate processors)
    i2i = glb_idx.get(FactorsRelationScaleObservation.partial_key())
    for s in i2i:
        octx = s.origin.processor.system
        dctx = s.destination.processor.system
        systems[octx][Factor].add(s.source_factor)
        systems[dctx][Factor].add(s.target_factor)
        systems[octx][FactorsRelationScaleObservation].add(s)
        systems[dctx][FactorsRelationScaleObservation].add(s)
        # TODO Expressions, AST, find parameters

    # All FLOW relationships
    for fr in glb_idx.get(FactorsRelationDirectedFlowObservation.partial_key()):
        octx = fr.source_factor.processor.system
        dctx = fr.target_factor.processor.system
        systems[octx][Factor].add(fr.source_factor)
        systems[octx][Factor].add(fr.target_factor)  # Target must be must be considered as a node in the same context!!
        systems[dctx][Factor].add(fr.target_factor)
        systems[dctx][Factor].add(fr.source_factor)  # Origin must be must be considered as a node in the same context!!
        systems[octx][FactorsRelationDirectedFlowObservation].add(fr)
        systems[dctx][FactorsRelationDirectedFlowObservation].add(fr)
        # TODO Expressions, AST, find parameters

    # Elaborate observation structures
    observations_prd, oers, time_periods = get_observations(glb_idx, relative=False)
    rel_observations_prd, rel_oers, _ = get_observations(glb_idx, relative=True)

    # SCALES --------------------------

    # Obtain a i2i Scales Graph
    sg = prepare_scales_graph(i2i)

    # Obtain the scale VALUES
    scales_prd = get_scaled(global_parameters, problem_statement.scenarios, sg, observations_prd, time_periods)

    # FLOWS --------------------------
    for ctx in systems:
        # From Factors IN the context (LOCAL, ENVIRONMENT or OUTSIDE)
        # obtain a basic graph. Signal each Factor as LOCAL or EXTERNAL, and SOCIETY or ENVIRONMENT
        bg = prepare_interfaces_graph(systems[ctx][Factor])

        # Obtain a flow graph
        #
        fg = get_flow_graph(bg, systems[ctx][FactorsRelationDirectedFlowObservation])

        # TODO Expand flow graph with it2it transforms

        # Split flow graphs
        for scenario_idx, (scenario_name, scenario) in enumerate(problem_statement.scenarios.items()):
            params = evaluate_parameters_for_scenario(global_parameters, scenario)
            for time_period in time_periods:
                # Obtain quantities, separate by time.
                # Also, identify SCALES i2i, intraprocessor (from Observations)
                obs = observations_prd.get(dict(__t=time_periods[time_period]))
                scales = scales_prd.get(dict(__t=time_periods[time_period], __s=scenario_idx))
                # TODO Overwrite "obs" with "scales" results
                # TODO Put observations into the flow-graph
                put_
                # TODO Put processors into scale (intensive to extensive conversion)
                scale_unit_processors(fg, params, rel_observations_prd)

                for sub_fg in nx.weakly_connected_component_subgraphs(fg):
                    # TODO Elaborate information flow graph
                    #      Cycles allowed?
                    ifg = get_information_flow_graph(sub_fg)
                    # TODO Solve information flow graph. From all possible combinations:
                    #  bottom-up if top-down USE
                    #  bottom-up if top-down DO NOT USE
                    #  top-down  if bottom-up USE
                    #  top-down  if bottom-up DO NOT USE
                    solve_flow_graph(sub_fg, ifg)  # Each value: Interface, Scenario, Time, Given/Computed -> VALUE (or UNDEFINED)
                    # TODO Put results back
        # TODO INDICATORS --- (INSIDE FLOWS)

    return []

# ######################################################################################################################
# GRAPH PARTITIONING - SCENARIO (PARAMETERS), SINGLE SCENARIO PARTITION CATEGORIES, IN-SCENARIO OBSERVATIONS VARIATION
# ######################################################################################################################


def get_contexts(objects):
    """

    :param objects:
    :return:
    """
    pass


def get_graph_partitioning_categories(objects):
    """
    Obtain categories known to partition the graph (into components)

    The most important one is usually "GEO"

    :param objects:
    :return: A list of partition categories
    """
    return ["GEO"]  # TODO Ensure that processors get category "GEO"


# ######################################################################################################################
# PARAMETERS - SCENARIOS. Global parameters. Scenarios to make
# ######################################################################################################################

def get_parameters(objects):
    """
    Obtain all parameters
    Obtain the variation ranges (or constant values) for each parameter

    :param objects:
    :return: A list (or dict) of parameters
    """
    # TODO
    return None


def map_parameters(params, objects):
    """
    Scan all occurrences of parameters
    Put a list in each parameter pointing to the parameter occurrence (expressions)

    :param params:
    :param objects:
    :return:
    """


def get_observation_variation_categories(objects):
    """
    Obtain categories by which, for a given scenario and partition, ie, a decoupled subsystem
    produce variation in quantitative observations, not in the system structure.

    The most important ones are TIME and OBSERVER

    :param objects:
    :return:
    """
    return ["TIME", "SOURCE"]

# ######################################################################################################################
# GENERATORS
# ######################################################################################################################


def get_scenario_generator(params, objects):
    """
    Obtain a scenario generator
    For each iteration it will return a dict with parameter name as key and parameter value as value

    :param params:
    :param objects:
    :return:
    """


def get_partition_generator(scenario, partition_categories, objects):
    """
    Obtain an enumerator of partitions for a given scenario (the set may change from scenario to scenario)
    For each iteration it will return a list of dicts made category names and category values
    (the list may be just of an element)

    :param scenario:
    :param partition_categories:
    :param objects:
    :return:
    """


def get_obs_variation_generator(obs_variation_categories, msm):
    """
    Obtain an iterator on the possible combinations of observation variations, given a MuSIASEM model

    Ideally this would be between one and less than ten [1, 10)
    For each iteration it will return a dict of categories

    :param obs_variation_categories:
    :param msm:
    :return:
    """


# ######################################################################################################################
# DATA STRUCTURES
# ######################################################################################################################


def build_msm_from_parsed(scenario, partition, parse_execution_results):
    """
    Construct MuSIASEM entities given the scenario and the requested partition

    :param scenario:
    :param partition:
    :param parse_execution_results:
    :return:
    """
    # TODO Everything!!!
    # TODO Consider evaluation of parameters in expressions

    return None


def cleanup_unused_processors(msm):
    """
    Remove or disable processors not involved in the solving and indicators
    For instance, unit processors and its children

    Elaborating a mark signaling when we have a unit processor is needed, then filtering out is trivial

    :param msm:
    :return:
    """
    # TODO


def reset_msm_solution_observations(msm):
    """
    Because the solving process in a MuSIASEM model is iterative, and each iteration may add observations,
    a reset is needed before iterations begin.



    :param msm:
    :return:
    """


def put_solution_into_msm(msm, fg):
    """

    :param msm:
    :param fg:
    :return:
    """


def get_flow_graph(msm, obs_variation):
    """
    Obtain a flow graph from the MuSIASEM model
    Assign quantitative observations filtering using the observations variation parameter

    :param msm:
    :param obs_variation:
    :return:
    """
    # TODO
    return nx.DiGraph()


# ######################################################################################################################
# INDICATORS
# ######################################################################################################################


def get_empty_indicators_collector():
    """
    Prepare a data structure where indicators

    It may be an in memory structure or a handle to a persistent structure

    :return:
    """
    # TODO
    return dict()


def compute_local_indicators(msm):
    """
    Compute indicator local to processors

    The standard would be metabolic ratios, wherever it may be possible

    :param msm: Input and output MuSIASEM model, augmented with the local indicators, that would be attached to the Processors
    :return: <nothing>
    """


def compute_global_indicators(msm):
    """
    Compute global indicators -those using data from more than one Processor-
    Store results inside the MuSIASEM structure

    :param msm:
    :return: <Nothing>
    """


def collect_indicators(indicators, msm):
    """
    Extract categories, observations and indicators into the "indicators" analysis structure

    :param indicators:
    :param msm:
    :return: <Nothing>
    """


# ######################################################################################################################
# SOLVER
# ######################################################################################################################


def solve(g):
    """
    Given a flow graph, find the values for the unknown interfaces

    How to treat the situation when an interface, with given value, could get a solution?
    * We have a top-down value
    * A bottom-up value is also available
    * Take note of the solution?

    :param g:
    :return:
    """
    #
    # Elaborate square matrix?




def solver_one(state):
    """
    Solves a MuSIASEM case study AND computes/collects indicators

    Receives as input a registry of parsed/elaborated MuSIASEM objects

    STORES in "state" an indicators structure (a Dataset?) with all the results

    :param state:
    :return: A list of issues
    """

    # Obtain the different state elements
    glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)

    # Obtain parameters and their variation ranges (initially decoupled)
    params_list = get_parameters(state)

    # Map params to the objects where they appear
    map_parameters(params_list, state)  # Modifies "params_list"

    # Obtain "partition" categories
    # Partition categories are categories provoking totally decoupled systems. The most important one is GEO
    partition_categories = get_graph_partitioning_categories(state)

    # Obtain "observation variation" categories
    # Observation variation categories are those which for a given scenario and partition (i.e., fixed subsystem)
    # produce variation in quantitative observations, not in the system structure. The most important are TIME and OBSERVER
    obs_variation_categories = get_observation_variation_categories(state)

    # Empty indicators collector
    indicators = get_empty_indicators_collector()

    for scenario in get_scenario_generator(params_list, state):
        # "scenario" contains a list of parameters and their values
        for partition in get_partition_generator(scenario, partition_categories, state):
            # "partition" contains a list of categories and their specific values
            # Build MSM for the partition categories
            msm = build_msm_from_parsed(scenario, partition, state)
            # TODO Remove processors not in the calculations (unit processors)
            cleanup_unused_processors(msm)  # Modify "msm"
            for obs_variation in get_obs_variation_generator(obs_variation_categories, state):
                # TODO "obs_variation" contains a list of categories and their specific values
                # TODO Build flow graph with observations filtered according to "obs_variation".
                # Nodes keep link to interface AND a value if there is one.
                # Edges keep link to: hierarchy OR flow OR scale change (and context)
                reset_msm_solution_observations(msm)
                fg = get_flow_graph(msm, obs_variation)
                for sub_fg in nx.weakly_connected_component_subgraphs(fg):
                    # Solve the sub_fg. Attach solutions to Nodes of "sub_fg"
                    solve(sub_fg)  # Modify "fg"
                    put_solution_into_msm(msm, sub_fg)  # Modify "msm"
                compute_local_indicators(msm)
                compute_global_indicators(msm)
                collect_indicators(indicators, msm)  # Elaborate output matrices
    return indicators
