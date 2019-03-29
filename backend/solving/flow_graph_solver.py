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
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Any, Tuple, Union, Optional, NamedTuple, NoReturn

from backend import case_sensitive
from backend.command_generators.parser_ast_evaluators import ast_evaluator
from backend.command_generators.parser_field_parsers import string_to_ast, expression_with_parameters, is_year, is_month
from backend.common.helper import create_dictionary, PartialRetrievalDictionary, ifnull, Memoize
from backend.models.musiasem_concepts import ProblemStatement, Parameter, FactorsRelationDirectedFlowObservation, \
    FactorsRelationScaleObservation, Processor, FactorQuantitativeObservation, Factor, \
    ProcessorsRelationPartOfObservation, FactorType
from backend.model_services import get_case_study_registry_objects, State
from backend.models.musiasem_concepts_helper import find_quantitative_observations
from backend.solving.graph.computation_graph import ComputationGraph
from backend.solving.graph.flow_graph import FlowGraph, IType
from backend.models.statistical_datasets import Dataset, Dimension
from backend.command_generators import Issue


class SolvingException(Exception):
    pass


@Memoize
def get_processor_name(processor: Processor, registry: PartialRetrievalDictionary) -> str:
    """ Get the processor hierarchical name with caching enabled """
    full_name = processor.full_hierarchy_names(registry)[0]
    return full_name if case_sensitive else full_name.lower()


def get_interface_name(interface: Factor, registry: PartialRetrievalDictionary) -> str:
    """ Get the full interface name prefixing it with the processor hierarchical name """
    # TODO: use Interface Type name "interface.taxon.name" instead of Interface name "interface.name"?
    return get_processor_name(interface.processor, registry) + ":" + \
           (interface.name if case_sensitive else interface.name.lower())


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
    result_params.update({p.name: p.default_value for p in base_params if p.default_value is not None})

    # Overwrite with scenario expressions or constants
    result_params.update(scenario_params)

    state = State()
    known_params = create_dictionary()
    unknown_params = create_dictionary()

    # Now, evaluate ALL expressions
    for param, expression in result_params.items():
        value, ast, params, issues = evaluate_numeric_expression_with_parameters(expression, state)
        if value is None:  # It is not a constant, store the parameters on which this depends
            if case_sensitive:
                unknown_params[param] = (ast, set(params))
            else:
                unknown_params[param] = (ast, set([p.lower() for p in params]))
        else:  # It is a constant, store it
            result_params[param] = value  # Overwrite
            known_params[param] = value

    cycles = get_circular_dependencies(unknown_params)
    if len(cycles) > 0:
        raise SolvingException(
            IType.ERROR, f"Parameters cannot have circular dependencies. {len(cycles)} cycles were detected: "
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
                if value is None:
                    raise SolvingException(
                        IType.ERROR, f"It should be possible to evaluate the parameter '{param}'. "
                                     f"Issues: {', '.join(issues)}")
                else:
                    del unknown_params[param]
                    result_params[param] = value
                    state.set(param, value)

    if len(unknown_params) > 0:
        raise SolvingException(IType.ERROR, f"Could not evaluate the following parameters: {', '.join(unknown_params)}")

    return result_params


TimeObservationsType = Dict[str, List[Tuple[float, FactorQuantitativeObservation]]]


def get_observations_by_time(prd: PartialRetrievalDictionary) -> Tuple[TimeObservationsType, TimeObservationsType]:
    """
    Process All QQ observations (intensive or extensive):
    * Store in a compact way (then clear), by Time-period, by Interface, by Observer.
    * Convert to float or prepare AST
    * Store as value the result plus the QQ observation (in a tuple)

    :param prd:
    :return:
    """
    observations: Dict[str, List[Tuple[float, FactorQuantitativeObservation]]] = {}
    state = State()

    # Get all observations by time
    for observation in find_quantitative_observations(prd, processor_instances_only=True):

        # Try to evaluate the observation value
        value, ast, _, issues = evaluate_numeric_expression_with_parameters(observation.value, state)

        # Store: (Value, FactorQuantitativeObservation)
        time = observation.attributes["time"].lower()
        if time not in observations:
            observations[time] = []

        observations[time].append((ifnull(value, ast), observation))

    if len(observations) == 0:
        return {}, {}

    # Check all time periods are consistent. All should be Year or Month, but not both.
    time_period_type = get_type_from_all_time_periods(list(observations.keys()))
    assert(time_period_type in ["year", "month"])

    # Remove generic period type and insert it into all specific periods. E.g. "Year" into "2010", "2011" and "2012"
    if time_period_type in observations:
        # Generic monthly ("Month") or annual ("Year") data
        periodic_observations = observations[time_period_type]
        del observations[time_period_type]

        for time in observations:
            observations[time] += periodic_observations

    return split_observations_by_relativeness(observations)


def evaluate_numeric_expression_with_parameters(expression: Union[float, str, dict], state: State) \
        -> Tuple[Optional[float], Optional[Dict], Set, List[str]]:

    issues: List[Tuple[int, str]] = []
    ast = None
    value = None
    params = set()

    if expression is None:
        value = None

    elif isinstance(expression, float):
        value = expression

    elif isinstance(expression, dict):
        ast = expression
        value, params = ast_evaluator(ast, state, None, issues)
        if value is not None:
            ast = None

    elif isinstance(expression, str):
        try:
            value = float(expression)
        except ValueError:
            ast = string_to_ast(expression_with_parameters, expression)
            value, params = ast_evaluator(ast, state, None, issues)
            if value is not None:
                ast = None

    else:
        issues.append((3, f"Invalid type '{type(expression)}' for expression '{expression}'"))

    return value, ast, params, [i[1] for i in issues]


def get_type_from_all_time_periods(time_periods: List[str]) -> Optional[str]:
    """ Check if all time periods are of the same period type, either Year or Month:
         - general "Year" & specific year (YYYY)
         - general "Month" & specific month (mm-YYYY or YYYY-mm, separator can be any of "-/")

    :param time_periods:
    :return:
    """
    # Based on the first element we will check the rest of elements
    period = next(iter(time_periods))

    if period == "year" or is_year(period):
        period_type = "year"
        period_check = is_year
    elif period == "month" or is_month(period):
        period_type = "month"
        period_check = is_month
    else:
        return None

    for time_period in time_periods:
        if time_period != period_type and not period_check(time_period):
            return None

    return period_type


def split_observations_by_relativeness(observations_by_time: TimeObservationsType):
    observations_by_time_norelative = {}
    observations_by_time_relative = {}
    for time, observations in observations_by_time.items():
        observations_by_time_norelative[time] = []
        observations_by_time_relative[time] = []
        for value, obs in observations:
            if obs.is_relative:
                observations_by_time_relative[time].append((value, obs))
            else:
                observations_by_time_norelative[time].append((value, obs))

    return observations_by_time_norelative, observations_by_time_relative


def compute_all_graph_combinations(comp_graph: ComputationGraph, params: Dict[str, float]) -> Dict[frozenset, Dict[str, float]]:
    all_values: Dict[frozenset, Dict[str, float]] = {}

    print(f"****** NODES: {comp_graph.nodes}")

    # Obtain nodes without a value
    compute_nodes = [n for n in comp_graph.nodes if params.get(n) is None]

    # Compute the missing information with the computation graph
    if len(compute_nodes) == 0:
        print("All nodes have a value. Nothing to solve.")
        return {}

    print(f"****** UNKNOWN NODES: {compute_nodes}")
    print(f"****** PARAMS: {params}")

    conflicts = comp_graph.compute_param_conflicts(set(params.keys()))

    for s, (param, values) in enumerate(conflicts.items()):
        print(f"Conflict {s + 1}: {param} -> {values}")

    combinations = ComputationGraph.compute_param_combinations(conflicts)

    for s, combination in enumerate(combinations):
        print(f"Combination {s}: {combination}")

        filtered_params = {k: v for k, v in params.items() if k in combination}
        results, _ = comp_graph.compute_values(compute_nodes, filtered_params)

        results_with_values = {k: v for k, v in results.items() if v is not None}
        print(f'  results_with_values={results_with_values}')
        print(f'  results_without_values={[k for k, v in results.items() if v is None]}')

        all_values[combination] = results_with_values

    return all_values


def compute_graph_values(comp_graph: ComputationGraph, params: Dict[str, float], other_values: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    print(f"****** NODES: {comp_graph.nodes}")

    # Filter params in graph
    graph_params = {k: v for k, v in params.items() if k in comp_graph.nodes}
    print("Missing values: ", [k for k, v in graph_params.items() if v is None])

    conflicts = comp_graph.compute_param_conflicts(set(graph_params.keys()))

    conflict_strings: List[str] = []
    for param, conf_params in conflicts.items():
        if len(conf_params) > 0:
            conf_params_string = "{" + ', '.join([f"{p} ({round(graph_params[p], 3)})" for p in conf_params]) + "}"
            conflict_strings.append(f"{param} ({round(graph_params[param], 3)}) -> {conf_params_string}")

    if len(conflict_strings) > 0:
        raise SolvingException(IType.ERROR, f"There are conflicts: {', '.join(conflict_strings)}")

    graph_params = {**graph_params, **other_values}

    # Obtain nodes without a value
    compute_nodes = [n for n in comp_graph.nodes if graph_params.get(n) is None]

    # Compute the missing information with the computation graph
    if len(compute_nodes) == 0:
        print("All nodes have a value. Nothing to solve.")
        return {}, []

    print(f"****** UNKNOWN NODES: {compute_nodes}")
    print(f"****** PARAMS: {graph_params}")

    results, _ = comp_graph.compute_values(compute_nodes, graph_params)

    results_with_values: Dict[str, float] = {}
    unknown_nodes: List[str] = []
    for k, v in results.items():
        if v is not None:
            results_with_values[k] = v
        else:
            unknown_nodes.append(k)

    return results_with_values, unknown_nodes


def split_name(processor_interface: str) -> Tuple[str, Optional[str]]:
    l = processor_interface.split(":")
    if len(l) > 1:
        return l[0], l[1]
    else:
        return l[0], None


class Edge(NamedTuple):
    src: Factor
    dst: Factor
    weight: Optional[str]


def add_factor_edges(glb_idx, graph: nx.DiGraph, musiasem_class: type, attributes: Tuple[str, str, str]):
    edges: List[Tuple[Factor, Factor, Optional[str]]] = \
        [(getattr(r, attributes[0]), getattr(r, attributes[1]), getattr(r, attributes[2]))
         for r in glb_idx.get(musiasem_class.partial_key())]

    for src, dst, weight in edges:
        src_name = get_interface_name(src, glb_idx)
        dst_name = get_interface_name(dst, glb_idx)
        if "Archetype" in [src.processor.instance_or_archetype, dst.processor.instance_or_archetype]:
            print(f"WARNING: excluding relation from '{src_name}' to '{dst_name}' because of Archetype processor")
        else:
            graph.add_edge(src_name, dst_name, weight=weight)


def resolve_weight_expressions(graph: nx.DiGraph, state: State, raise_error=False) -> NoReturn:
    for u, v, data in graph.edges(data=True):
        expression = data["weight"]
        if expression is not None:
            value, ast, params, issues = evaluate_numeric_expression_with_parameters(expression, state)
            if raise_error and value is None:
                raise SolvingException(IType.ERROR,
                    f"Cannot evaluate expression "
                    f"'{expression}' for weight from interface '{u}' to interface '{v}'. Params: {params}. "
                    f"Issues: {', '.join(issues)}"
                )

            data["weight"] = ifnull(value, ast)


def compute_flow_results(state: State, glb_idx, global_parameters, problem_statement):
    # Get all interface observations. Also resolve expressions without parameters. Cannot resolve expressions
    # depending only on global parameters because some of them can be overridden by scenario parameters.
    time_observations_absolute, time_observations_relative = get_observations_by_time(glb_idx)

    if len(time_observations_absolute) == 0:
        raise SolvingException(
            IType.WARNING, f"No absolute observations have been found. The solver has nothing to solve."
        )

    # Add Interfaces -Flow- relations (time independent)
    relations_flow = nx.DiGraph()
    add_factor_edges(glb_idx, relations_flow, FactorsRelationDirectedFlowObservation, ("source_factor", "target_factor", "weight"))

    # Add Processors -Scale- relations (time independent)
    relations_scale = nx.DiGraph()
    add_factor_edges(glb_idx, relations_scale, FactorsRelationScaleObservation, ("origin", "destination", "quantity"))

    # TODO Expand flow graph with it2it transforms
    # relations_scale_it2it = glb_idx.get(FactorTypesRelationUnidirectionalLinearTransformObservation.partial_key())

    # First pass to resolve weight expressions: only expressions without parameters can be solved
    resolve_weight_expressions(relations_flow, state)
    resolve_weight_expressions(relations_scale, state)

    results: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    for scenario_name, scenario_params in problem_statement.scenarios.items():  # type: str, dict
        print(f"********************* SCENARIO: {scenario_name}")

        scenario_state = State(evaluate_parameters_for_scenario(global_parameters, scenario_params))

        for time_period, observations in time_observations_absolute.items():
            print(f"********************* TIME PERIOD: {time_period}")

            # Final values are taken from "observations" that need to computed
            known_observations = {}
            # Create a copy of the main relations structure that is modified with time-dependent values
            time_relations_flow = relations_flow.copy()
            time_relations_scale = relations_scale.copy()

            # Second and last pass to resolve observation expressions with parameters
            for expression, obs in observations:
                interface_name = get_interface_name(obs.factor, glb_idx)
                if interface_name not in time_relations_flow.nodes and interface_name not in time_relations_scale.nodes:
                    print(f"WARNING: observation at interface '{interface_name}' is not taken into account.")
                else:
                    value, _, params, issues = evaluate_numeric_expression_with_parameters(expression, scenario_state)
                    if value is None:
                        raise SolvingException(IType.ERROR,
                            f"Scenario '{scenario_name}' - period '{time_period}'. Cannot evaluate expression "
                            f"'{expression}' for observation at interface '{interface_name}'. Params: {params}. "
                            f"Issues: {', '.join(issues)}"
                        )

                    known_observations[interface_name] = value

            assert(len(known_observations) > 0)

            # Add Processors internal -RelativeTo- relations (time dependent)
            # Transform relative observations into graph edges
            for expression, obs in time_observations_relative[time_period]:
                processor_name = get_processor_name(obs.factor.processor, glb_idx)
                time_relations_scale.add_edge(
                    processor_name + ":" + (obs.relative_factor.name if case_sensitive else obs.relative_factor.name.lower()),
                    processor_name + ":" + (obs.factor.name if case_sensitive else obs.factor.name.lower()),
                    weight=expression)

            # Second and last pass to resolve weight expressions: expressions with parameters can be solved
            resolve_weight_expressions(time_relations_flow, scenario_state, raise_error=True)
            resolve_weight_expressions(time_relations_scale, scenario_state, raise_error=True)

            # if scenario_name == 'Scenario1' and time_period == '2011':
            #     for graph in [time_relations_scale]:  # time_relations_flow
            #         for component in nx.weakly_connected_components(graph):
            #             # plt.figure(1, figsize=(8, 8))
            #             nx.draw_spring(graph.subgraph(component), with_labels=True, font_size=8, node_size=60)
            #             # nx.draw_kamada_kawai(graph.subgraph(component), with_labels=True)
            #             plt.show()


            # *********************************************
            flow_graph = FlowGraph(time_relations_flow)
            comp_graph_flow, issues = flow_graph.get_computation_graph(time_relations_scale)

            for issue in issues:
                print(issue)

            error_issues = [e.description for e in issues if e.itype == IType.ERROR]
            if len(error_issues) > 0:
                raise SolvingException(
                    IType.ERROR,
                    f"Scenario '{scenario_name}' - period '{time_period}'. The computation graph cannot "
                    f"be generated. Issues: {', '.join(error_issues)}"
                )

            # *********************************************
            comp_graph_scale = ComputationGraph(time_relations_scale)

            # *********************************************

            results[(scenario_name, time_period, "0")] = {**known_observations}

            current_num_unknown_nodes = len(comp_graph_flow.nodes)
            previous_num_unknown_nodes = current_num_unknown_nodes + 1
            other_values1 = {}
            other_values2 = {}
            params1 = None
            params2 = None
            data = {}
            while previous_num_unknown_nodes > current_num_unknown_nodes > 0:
                previous_num_unknown_nodes = current_num_unknown_nodes
                if params1 is None:
                    params1 = {**data, **known_observations}
                else:
                    params1 = {**data}
                data, unknown_nodes = compute_graph_values(comp_graph_scale, params1, other_values1)
                results[(scenario_name, time_period, "0")].update(data)
                other_values1.update({**data, **params1})
                if params2 is None:
                    params2 = {**data, **known_observations}
                else:
                    params2 = {**data}
                data, unknown_nodes = compute_graph_values(comp_graph_flow, params2, other_values2)
                results[(scenario_name, time_period, "0")].update(data)
                other_values2.update({**data, **params2})
                current_num_unknown_nodes = len(unknown_nodes)

            # TODO INDICATORS

    return results


def compute_interfacetype_aggregates(glb_idx, results):

    def get_sum(processor: str, children: Set[FactorType]) -> float:
        sum_children = 0.0
        for child in children:
            child_name = child.name if case_sensitive else child.name.lower()
            child_value = values.get(processor + ":" + child_name)
            if child_value is None:
                if child_name in parent_interfaces:
                    child_value = get_sum(proc, parent_interfaces[child_name])
                else:
                    child_value = 0.0

            sum_children += child_value

        return sum_children

    agg_results: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    processors: Set[str] = {split_name(proc_interface)[0] for values in results.values() for proc_interface in values}

    # Get all different existing interfaces types that can be computed based on children interface types
    parent_interfaces: Dict[str, Set[FactorType]] = \
        {i.name if case_sensitive else i.name.lower(): i.get_children()
         for i in glb_idx.get(FactorType.partial_key()) if len(i.get_children()) > 0}

    for key, values in results.items():
        for parent_interface, children_interfaces in parent_interfaces.items():
            for proc in processors:
                proc_interface_name = proc + ":" + parent_interface

                if values.get(proc_interface_name) is None:
                    agg_results.setdefault(key, {}).update({
                        proc_interface_name: get_sum(proc, children_interfaces)
                    })

    return agg_results


def get_processor_partof_hierarchies(glb_idx, system):
    # Handle Processors -PartOf- relations
    proc_hierarchies = nx.DiGraph()

    # Just get the -PartOf- relations of the current system
    part_of_relations = [(r.child_processor, r.parent_processor)
                         for r in glb_idx.get(ProcessorsRelationPartOfObservation.partial_key())
                         if system in [r.parent_processor.processor_system, r.child_processor.processor_system]]

    for child_processor, parent_processor in part_of_relations:  # type: Processor, Processor
        child_name = get_processor_name(child_processor, glb_idx)
        parent_name = get_processor_name(parent_processor, glb_idx)

        # Parent and child should be of the same system
        assert (child_processor.processor_system == parent_processor.processor_system)

        if "Archetype" in [parent_processor.instance_or_archetype, child_processor.instance_or_archetype]:
            print(f"WARNING: excluding relation from '{child_name}' to '{parent_name}' because of Archetype processor")
        else:
            proc_hierarchies.add_edge(child_name, parent_name, weight=1.0)

    return proc_hierarchies


def compute_partof_aggregates(glb_idx, systems, results):
    agg_results: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    # agg_combinations: Dict[frozenset, str] = {}

    # Get all different existing interfaces and their units
    # TODO: interfaces could need a unit transformation according to interface type
    interfaces: Set[str] = {i.name if case_sensitive else i.name.lower() for i in glb_idx.get(FactorType.partial_key())}

    for system in systems:
        print(f"********************* SYSTEM: {system}")

        proc_hierarchies = get_processor_partof_hierarchies(glb_idx, system)

        # Iterate over all independent trees created by the PartOf relations
        for component in nx.weakly_connected_components(proc_hierarchies):
            proc_hierarchy: nx.DiGraph = proc_hierarchies.subgraph(component)
            print(f"********************* HIERARCHY: {proc_hierarchy.nodes}")

            # plt.figure(1, figsize=(8, 8))
            # nx.draw_spring(proc_hierarchy, with_labels=True, font_size=8, node_size=60)
            # plt.show()

            for interface_name in interfaces:
                print(f"********************* INTERFACE: {interface_name}")

                interfaced_proc_hierarchy = nx.DiGraph(
                    incoming_graph_data=[(u+":"+interface_name, v+":"+interface_name) for u, v in proc_hierarchy.edges]
                )

                for key, values in results.items():

                    filtered_values = {k: values[k] for k in interfaced_proc_hierarchy.nodes
                                       if values.get(k) is not None}

                    if len(filtered_values) > 0:
                        print(f"*** (Scenario, Period, Combination = {key}")

                        # Resolve computation graph
                        # comp_graph = ComputationGraph()
                        # comp_graph.add_edges(list(interfaced_proc_hierarchy.edges), 1.0, None)
                        # agg_res = compute_all_graph_combinations(comp_graph, filtered_results)
                        #
                        # for comb, data in agg_res.items():
                        #     if len(data) > 0:
                        #         if agg_combinations.get(comb) is None:
                        #             agg_combinations[comb] = str(len(agg_combinations))
                        #
                        #         agg_results[(scenario_name, time_period, f"({combination}, {agg_combinations[comb]})")] = data

                        # Aggregate top-down, starting from root
                        agg_res, error_msg = compute_aggregate_results(interfaced_proc_hierarchy, filtered_values)

                        if error_msg is not None:
                            raise SolvingException(
                                IType.ERROR, f"System: '{system}'. Interface: '{interface_name}'. Error: {error_msg}")

                        if len(agg_res) > 0:
                            agg_results.setdefault(key, {}).update(agg_res)

    return agg_results, {}


def flow_graph_solver(global_parameters: List[Parameter], problem_statement: ProblemStatement,
                      input_systems: Dict[str, Set[Processor]], state: State) -> List[Issue]:
    """
    A solver

    :param global_parameters: Parameters including the default value (if defined)
    :param problem_statement: ProblemStatement object, with scenarios (parameters changing the default)
                              and parameters for the solver
    :param state: State with everything
    :param input_systems: A dictionary of the different systems to be solved
    :return: List of Issues
    """
    glb_idx, _, _, datasets, _ = get_case_study_registry_objects(state)

    try:
        results = compute_flow_results(state, glb_idx, global_parameters, problem_statement)

        agg_results, agg_combinations = compute_partof_aggregates(glb_idx, input_systems, results)

    except SolvingException as e:
        return [Issue(e.args[0], str(e.args[1]))]

    # Add "agg_results" to "results"
    for key, value in agg_results.items():
        results[key].update(value)

    agg_results = compute_interfacetype_aggregates(glb_idx, results)

    # Add "agg_results" to "results"
    for key, value in agg_results.items():
        results[key].update(value)

    def create_dataframe(r: Dict[Tuple[str, str, str], Dict[str, float]]) -> pd.DataFrame:
        data = {k + split_name(name): {"Value": value}
                for k, v in r.items()
                for name, value in v.items()}
        return pd.DataFrame.from_dict(data, orient='index')

    df = create_dataframe(results)

    # Round all values to 3 decimals
    df = df.round(3)

    # Give a name to the dataframe indexes
    df.index.names = ["Scenario", "Period", "Combination", "Processor", "Interface"]

    # Sort the dataframe based on indexes. Not necessary, only done for debugging purposes.
    df = df.sort_index(level=["Scenario", "Period", "Combination", "Processor", "Interface"])

    # Adding column with interface units
    interface_units = create_dictionary(data={i.name: i.attributes.get('unit')
                                              for i in glb_idx.get(FactorType.partial_key())})
    df["Unit"] = [interface_units[i] for i in df.index.get_level_values("Interface")]

    # Add customer attribute "level"
    processors = create_dictionary(data={get_processor_name(p, glb_idx): p.attributes.get("level", "")
                                         for p in glb_idx.get(Processor.partial_key())})
    df["Level"] = [processors[p] for p in df.index.get_level_values("Processor")]
    df.set_index("Level", append=True, inplace=True)

    print(df)

    # Create dataset and store in State
    datasets["flow_graph_solution"] = get_dataset(df)

    # Create dataset and store in State
    datasets["end_use_matrix"] = get_eum_dataset(df)

    return []


def compute_aggregate_results(tree: nx.DiGraph, params: Dict[str, float]) -> Tuple[Dict[str, float], Optional[str]]:
    def compute_node(node: str) -> float:
        if params.get(node) is not None:
            return params[node]

        sum_children = 0
        for pred in tree.predecessors(node):
            sum_children += compute_node(pred)

        values[node] = sum_children
        return sum_children

    root_nodes = [node for node, degree in tree.out_degree() if degree == 0]
    if len(root_nodes) != 1 or root_nodes[0] is None:
        return {}, f"Root node cannot be taken from list '{root_nodes}'"

    values: Dict[str, float] = {}
    compute_node(root_nodes[0])
    return values, None


def get_eum_dataset(dataframe: pd.DataFrame) -> "Dataset":
    # EUM columns
    df = dataframe.query('Interface in ["biofuel", "cropproduction", "fertilizer", "ha", "lu"]')

    # EUM rows
    df = df.query('Processor in ['
                  '"society", "society.biodiesel", "society.bioethanol", "society.commerceimports", "society.commerceexports", '
                  '"society.bioethanol.cereals", '
                  '"society.bioethanol.cereals.wheat", "society.bioethanol.cereals.maize", '
                  '"society.bioethanol.cereals.externalwheat", "society.bioethanol.cereals.externalmaize", '
                  '"society.bioethanol.sugarcrops", '
                  '"society.bioethanol.sugarcrops.sugarbeet", "society.bioethanol.sugarcrops.sugarcane", '
                  '"society.bioethanol.sugarcrops.externalsugarbeet", "society.bioethanol.sugarcrops.externalsugarcane", '
                  '"society.biodiesel.oilcrops", '
                  '"society.biodiesel.oilcrops.palmoil", "society.biodiesel.oilcrops.rapeseed", "society.biodiesel.oilcrops.soybean", '
                  '"society.biodiesel.oilcrops.externalpalmoil", "society.biodiesel.oilcrops.externalrapeseed", "society.biodiesel.oilcrops.externalsoybean"'
                  ']')

    # df = df.query('Processor in ['
    #               '"society", "biodiesel", "bioethanol", "commerceimports", "commerceexports", '
    #               '"bioethanol.cereals", '
    #               '"bioethanol.cereals.wheat", "bioethanol.cereals.maize", '
    #               '"bioethanol.cereals.externalwheat", "bioethanol.cereals.externalmaize", '
    #               '"bioethanol.sugarcrops", '
    #               '"bioethanol.sugarcrops.sugarbeet", "bioethanol.sugarcrops.sugarcane", '
    #               '"bioethanol.sugarcrops.externalsugarbeet", "bioethanol.sugarcrops.externalsugarcane", '
    #               '"biodiesel.oilcrops", '
    #               '"biodiesel.oilcrops.palmoil", "biodiesel.oilcrops.rapeseed", "biodiesel.oilcrops.soybean", '
    #               '"biodiesel.oilcrops.externalpalmoil", "biodiesel.oilcrops.externalrapeseed", "biodiesel.oilcrops.externalsoybean"'
    #               ']')

    df = df.pivot_table(values="Value", index=["Scenario", "Period", "Processor", "Level"], columns="Interface")

    # Adding units to column name
    # TODO: remove hardcoded
    df = df.rename(columns={"biofuel": "Biofuel (tonnes)",
                            "cropproduction": "CropProduction (tonnes)",
                            "fertilizer": "Fertilizer (kg)",
                            "ha": "HA (h)",
                            "lu": "LU (ha)"})

    return get_dataset(df)


def get_dataset(dataframe: pd.DataFrame) -> "Dataset":
    ds = Dataset()
    ds.data = dataframe.reset_index()
    ds.code = "flow_graph_solution"
    ds.description = "Solution given by the Flow Graph Solver"
    ds.attributes = {}
    ds.metadata = None
    ds.database = None

    for dimension in dataframe.index.names:  # type: str
        d = Dimension()
        d.code = dimension
        d.description = None
        d.attributes = None
        d.is_time = (dimension.lower() == "period")
        d.is_measure = False
        d.dataset = ds

    for measure in dataframe.columns.values:  # type: str
        d = Dimension()
        d.code = measure
        d.description = None
        d.attributes = None
        d.is_time = False
        d.is_measure = True
        d.dataset = ds

    return ds
