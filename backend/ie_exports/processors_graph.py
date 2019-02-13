import networkx as nx

from backend.solving import *


def construct_processors_graph(state: State, query: IQueryObjects, filt: Union[str, dict]):
    """
    Prepare a graph representing processors and their interconnections
    Optionally represent the relation "part-of" between Processors

    :param state: State
    :param query: A IQueryObjects instance (which has been already injected the state)
    :param filt: A filter to be passed to the query instance
    :return:
    """
    # Format
    a_processor = dict(graphics={"type": "hexagon", "color": "#aa2211"})
    edge_flow = dict(graphics={"fill": "#ff00ff", "width": 3, "targetArrow": "standard"})
    edge_part_of = dict(graphics={"fill": "#00ffff", "width": 3, "targetArrow": "standard"})

    # Obtain the information needed to elaborate the graph
    objs = query.execute([Processor, Factor,
                          ProcessorsRelationPartOfObservation,
                          FactorsRelationDirectedFlowObservation
                          ],
                         filt
                         )

    # PartialRetrievalDictionary
    reg = state.get("_glb_idx")

    n = []
    e = []
    # Processors
    for p in objs[Processor]:
        # nam = " ,".join(p.full_hierarchy_names(reg))
        # print(nam)

        p_id = get_processor_ident(p)
        n.append((p_id, processor_to_dict(p, reg)))

    # Flow Links
    for flow in objs[FactorsRelationDirectedFlowObservation]:
        sf = get_factor_type_id(flow.source_factor)
        tf = get_factor_type_id(flow.target_factor)
        weight = flow.weight if flow.weight else ""
        weight = ("\n" + str(weight)) if weight != "" else ""
        sp_id = get_processor_ident(flow.source_factor.processor)
        dp_id = get_processor_ident(flow.target_factor.processor)
        if sf == tf:
            edge = dict(font={"size": 7}, label=sf + weight, w=str(weight))
        else:
            edge = dict(font={"size": 7}, label=sf + weight + "\n" + tf, w=str(weight))
        edge.update(edge_flow)
        e.append((sp_id, dp_id, edge))

    # Part-of Relations
    for po in objs[ProcessorsRelationPartOfObservation]:
        # TODO add edge between two Processors
        pp = get_processor_ident(po.parent_processor)
        cp = get_processor_ident(po.child_processor)
        edge = dict(font={"size": 7})
        edge.update(edge_part_of)
        e.append((cp, pp, edge))

    # NetworkX
    # -- Create the graph
    processors_graph = nx.MultiDiGraph()
    processors_graph.add_nodes_from(n)
    processors_graph.add_edges_from(e)

    # Convert to VisJS
    ids_map = create_dictionary()
    id_count = 0
    for node in processors_graph.nodes(data=True):
        sid = str(id_count)
        node[1]["id"] = sid
        ids_map[node[0]] = sid
        id_count += 1

    vis_nodes = []
    vis_edges = []
    for node in processors_graph.nodes(data=True):
        d = dict(id=node[1]["id"], label=node[1]["uname"])
        if "shape" in node[1]:
            # circle, ellipse, database, box, diamond, dot, square, triangle, triangleDown, text, star
            d["shape"] = node[1]["shape"]
        else:
            d["shape"] = "hexagon"
        if "color" in node[1]:
            d["color"] = node[1]["color"]
        vis_nodes.append(d)
    for edge in processors_graph.edges(data=True):
        f = ids_map[edge[0]]
        t = ids_map[edge[1]]
        d = {"from": f, "to": t, "arrows": "to"}
        data = edge[2]
        if "label" in data:
            d["label"] = data["label"]
            d["font"] = {"align": "middle"}
            if "font" in data:
                d["font"].update(data["font"])

        vis_edges.append(d)
    visjs = {"nodes": vis_nodes, "edges": vis_edges}
    # print(visjs)
    return visjs


if __name__ == '__main__':
    from backend.restful_service.serialization import deserialize_state

    # Deserialize previously recorded Soslaires State (WARNING! Execute unit tests to generated the ".serialized" file)
    fname = "/home/rnebot/GoogleDrive/AA_MAGIC/Soslaires.serialized"
    fname = "/home/rnebot/GoogleDrive/AA_MAGIC/MiniAlmeria.serialized"
    with open(fname, "r") as file:
        s = file.read()
    state = deserialize_state(s)
    # Create a Query and execute a query
    query = BasicQuery(state)
    print(construct_processors_graph(state, query, None))
