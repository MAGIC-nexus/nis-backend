#!/usr/bin/python3
# -*- coding=utf-8 -*-
#
"""
Creation: September 22th, 2016

@author: rnebot

Core entities able representing a MuSIASEM case study

Steps in building the model:
* Construct Processors hierarchy
* Construct FlowFund hierarchies
* Connect Processors with FlowFunds
* Define values
* Solve

Also:
Transforms / maps / functions, to convert FlowFund's
Metabolic Rates
Visualization: Processor trees, FlowFund trees,
Cube definitions? map

"""

import numpy as np


import pandas as pd
from enum import Enum


class FlowFundRoegenType(Enum):  # Used in FlowFund
    flow = 1
    fund = 0


class FlowFundBranchType(Enum):  # Used in FlowFund
    taxonomy = 1
    scales = 0


class HierarchyNode:
    def __init__(self, parent, name):
        self.name = name
        self.children = []
        self.parent = None
        self.set_parent(parent)

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


class FlowFund(HierarchyNode):
    """ FlowFund allows accounting for some asset (Fund) or flow
    FlowFund is recurrent, allowing the definition of a tree (or multitree if there are different scales)
    Roegen type is can be "Flow" or "Fund"
    The node points to its children, and also to its parent. Both can be empty: leave nodes will have no children, root
    node will have no parent. An isolated node will not have children not parent.
    The type definition specified if -extensive- values will be scalars, tuples or other things
    """
    def __init__(self, parent=None, name=None, roegen_type=FlowFundRoegenType.flow, is_waste_flow=False, dtype=np.float, branch_type=FlowFundBranchType.taxonomy):
        super().__init__(parent, name)
        # Categories (of a same taxonomy); Scales (different units). Only first level should be "Scales".
        self.children_branch_type = branch_type
        self.roegen_type = roegen_type  # "Flow" or "Fund". All nodes of the tree share the same Roegen type
        self.type_definition = dtype  # A valid Numpy dtype specifier
        self.is_waste_flow = is_waste_flow
        self.extension = {}  # Amount of FlowFund accounted per Processor (not for the FlowFund children)
        self.extension_of_children = None  # Total FlowFund accounted for children of the node (not for itself)
        self.links = []  # "Link" instances pointing to this specific FlowFund

    def set_roegen_type(self, roegen_type: FlowFundRoegenType, recurse: True):
        """ Set the roegen_type for the node, and optionally recurse to its descendants """
        self.roegen_type = roegen_type
        if recurse:
            for c in self.children:
                c.set_roegen_type(roegen_type)

    def set_type_definition(self, dtype: np.dtype, recurse: True):
        """ Set the dtype for values in links associated with this FlowFund node, and optionally recurse to its
        descendants """
        self.type_definition = dtype
        if recurse:
            for c in self.children:
                c.set_type_definition(dtype)

    def add_links(self, links: list, overwrite=False):
        """ Add links pointing to the current node. This allows tracking processor interconnections "charging" to
        the FlowFund at stake
        """
        # TODO Check that no double imputation has been specified

        # Check that all "links" are instances of "ProcessorPortsLink"
        for c in links:
            if type(c) is not ProcessorPortsLink:
                raise Exception("All children should be instances of 'ProcessorPortsLink'."
                                " Instance of '" + str(type(c)) + "' found.")

        # Change the children list (cloning)
        if overwrite:
            self.links = [c for c in links]
        else:
            self.links.extend(links)

        # Avoid duplicate links, removing them
        tmp = set([])
        to_delete = []
        for i, c in enumerate(self.links):
            if c in tmp:
                to_delete.append(i)
            else:
                tmp.add(c)
        for i in reversed(to_delete):
            del self.links[i]

        # Ensure links are pointing to the FundFlow
        for c in self.links:
            c.flow_fund = self


class ProcessorType(Enum):  # Used in Processor
    environment = 1
    target = 2
    external = 3


class Processor(HierarchyNode):
    """ Processor as defined by M. Giampietro's MuSIASEM. By virtue of attributes "parent" and "children" it can
    represent a holon, through a hierarchy of processors, which are part and whole, depending on the level of focus.
    Processors can be interconnected externally, with sibling processors -at the same level inside a Processor or inside
    the 'Tao', the None parent Processor-, but also intraconnected, with internal Processors. The objects allowing
    connections are "ports" """
    def __init__(self, parent=None, name=None, processor_type=ProcessorType.target):
        super().__init__(parent, name)
        self.processor_type = processor_type
        self.ports = []


class ProcessorPort:
    """ Port instances are associated with Processor in COMPOSITION relationship, so, if a Processsor ceases existing,
    its "ports" will also disappear """
    def __init__(self):
        self.name = None
        self.processor = None
        self.ff = None  # FlowFund type
        # The port part linking externally to other processors
        # It is a list (and not a single connection) to support split's (when outgoing) or join's (when incoming)
        self.outer_links = []
        # The port part linking to internal processors
        self.inner_links = []


class ProcessorPortsLink:
    def __init__(self,
                 source: ProcessorPort,
                 destination: ProcessorPort,
                 flow_fund: FlowFund,
                 extension: float=None,
                 name: str=None,
                 additional_information: str=None):
        self.name = name
        self.source = source  # Source Port
        self.destination = destination  # Destination port
        self.flow_fund = flow_fund  # FlowFund node specifying the type

        # Extensive value associated to the link, with type (scalar or tuple) in FlowFund node
        self.extension = extension
        # JSON for any information. For instance, how the value is obtained
        self.additional_information = additional_information


def _link(src: Processor, dst: Processor, ff: FlowFund, extension=None):
    """

    :param src:
    :param dst:
    :param ff:
    :param extension: The magnitude of the link
    :return: The list of links connecting "src" with "dst"
    """
    def find_nearest_common_ancestor():
        src_ancestors = []
        current = src
        while current:
            src_ancestors.append(current)
            current = current.parent
        src_ancestors.append(None)

        dst_ancestors = []
        current = dst
        while current:
            dst_ancestors.append(current)
            current = current.parent
        dst_ancestors.append(None)

        commons = set(src_ancestors).intersection(dst_ancestors)
        nearest = None
        for c in src_ancestors:
            if c in commons:
                nearest = c
                break

        if nearest:
            # Cut both ancestor lists
            i = 0
            curr = src_ancestors[i]
            tmp = [curr]
            while curr != nearest:
                i += 1
                curr = src_ancestors[i]
                tmp.append(curr)
            src_ancestors = tmp
            i = 0
            curr = dst_ancestors[i]
            tmp = [curr]
            while curr != nearest:
                i += 1
                curr = dst_ancestors[i]
                tmp.append(curr)
            dst_ancestors = tmp

        return nearest, src_ancestors, dst_ancestors

    def sublink(s: Processor, d: Processor):
        """
        Create a link between Processors, if there is no existing link already (with type "f")
        Two types of link (hierarchical and sibling)
        The link, which is returned, is registered to the FlowFund and in the Processors

        :param s: Source Processor
        :param d: Destination Processor
        :param f: FlowFund
        :return: The link
        """
        if s.parent == d.parent:
            # Create port in source and destination

            # Check if the link is not already created and if the new link will
            # not create a bidirectional port (not permitted)
            src_port = None
            for p in s.ports:
                if p.ff == ff:
                    src_port = p
                    for n in p.outer_links:
                        if n.source.processor != s:
                            raise Exception("A port is not bidirectional. Current source port "
                                            "is not source in all its outer links")
                        if n.destination.processor == d:
                            return n  # There is a link already

            # Check if the link is not already created and if the new link will
            # not create a bidirectional port (not permitted)
            dst_port = None
            for p in d.ports:
                if p.ff == ff:
                    dst_port = p
                    for n in p.outer_links:
                        if n.destination.processor != d:
                            raise Exception("A port is not bidirectional. Current destination port "
                                            "is not destination in all its outer links")
                        if n.source.processor == s:
                            return n  # There is a link already

            if not src_port:
                src_port = ProcessorPort()
                src_port.processor = s
                s.ports.append(src_port)
                src_port.ff = ff

            if not dst_port:
                dst_port = ProcessorPort()
                dst_port.processor = d
                d.ports.append(dst_port)
                dst_port.ff = ff

            # Link siblings
            lnk = ProcessorPortsLink(src_port, dst_port, ff, extension)
            src_port.outer_links.append(lnk)
            dst_port.outer_links.append(lnk)
            ff.add_links([lnk])

        elif s.parent == d or d.parent == s:
            # Link ancestor-descendant
            if s.parent == d:
                pa = d
                ch = s
            else:
                pa = s
                ch = d

            # Check for duplicate in parent Processor
            pa_port = None
            for p in pa.ports:
                if p.ff == ff:
                    pa_port = p
                    for n in p.inner_links:
                        if n.destination.processor == ch or n.source.processor == ch:
                            return n  # There is a link already

            # Check for duplicate in child Processor
            ch_port = None
            for p in ch.ports:
                if p.ff == ff:
                    ch_port = p
                    for n in p.outer_links:
                        if n.destination.processor == ch or n.source.processor == ch:
                            return n  # There is a link already

            if not pa_port:
                pa_port = ProcessorPort()
                pa_port.processor = pa
                pa.ports.append(pa_port)
                pa_port.ff = ff

            if not ch_port:
                ch_port = ProcessorPort()
                ch_port.processor = ch
                ch.ports.append(ch_port)
                ch_port.ff = ff

            if pa == s:
                src_port = pa_port
                dst_port = ch_port
            else:
                src_port = ch_port
                dst_port = pa_port

            # Link
            lnk = ProcessorPortsLink(src_port, dst_port, ff, extension)
            pa_port.inner_links.append(lnk)
            ch_port.outer_links.append(lnk)
            ff.add_links([lnk])

        else:
            raise Exception("Links should be between siblings or between immediate ancestor-descendant nodes")

        return lnk

    common_ancestor, src_ancestors, dst_ancestors = find_nearest_common_ancestor()

    lnk_lst = []

    if src != common_ancestor and dst != common_ancestor:  # Different branches, joined by two sibling Processors
        # Create links from Source to a child (direct descendant) of common_ancestor
        l_src = src
        while l_src != common_ancestor:
            l_dst = l_src.parent
            if l_dst != common_ancestor:
                lnk_lst.append(sublink(l_src, l_dst))
            l_src = l_dst

        rlst = reversed(dst_ancestors[:-1])
        tmp_lsrc = src_ancestors[-2]
        l_src = next(rlst)
        # Create a link between siblings
        lnk_lst.append(sublink(tmp_lsrc, l_src))

        # Create descending links down to Destination
        while l_src != dst:
            l_dst = next(rlst)
            lnk_lst.append(sublink(l_src, l_dst))
            l_src = l_dst

    else:  # A single branch, either move up or down
        if dst == common_ancestor:
            l_src = src
            while l_src != common_ancestor:
                l_dst = l_src.parent
                lnk_lst.append(sublink(l_src, l_dst))
                l_src = l_dst
        else:
            rlst = reversed(dst_ancestors)
            l_src = next(rlst)
            while l_src != dst:
                l_dst = next(rlst)
                lnk_lst.append(sublink(l_src, l_dst))
                l_src = l_dst

    return lnk_lst
    # TODO Check also that a FlowFund node is not repeated for the same Processor (repeated imputation!).
    # TODO A related higher level check could be assuring that no ancestor node is repeated


class CaseStudy:
    def __init__(self, name, root_processors, root_flowsfunds):
        self.name = name
        self.root_processors = root_processors
        self.root_flowsfunds = root_flowsfunds
        self.high_level_links = {}  # A registry of links. A list per FlowFund type
        self.low_level_links = {}  # A registry of all atomic links. A list per FlowFund type

    def link(self, src: Processor, dst: Processor, ff: FlowFund, extension=None):
        links = _link(src, dst, ff, extension)
        if ff not in self.high_level_links:
            self.high_level_links[ff] = []
        lst = self.high_level_links[ff]
        lst.append((src, dst, ff, extension, links))

    def clone_links_pattern(self, src_ff: FlowFund, dst_ff: FlowFund):
        """ Given the connection pattern of a FlowFund and its descendants, connect another FlowFund (member of other tree)
            to the same set of Processors
        """
        if src_ff in self.high_level_links:
            for t in self.high_level_links[src_ff]:
                self.link(t[0], t[1], dst_ff, None)

    def compute_bottom_up_flowfunds(self):
        visited_atomic_links = set([])  # A list of visited atomic links. Atomic links visited for the first time are reset (0)
        for ff in self.high_level_links:
            for lnk in self.high_level_links[ff]:
                # Source or Sink
                source = not ff.is_waste_flow
                if source:
                    if lnk[0] not in ff.extension:
                        ff.extension[lnk[0]] = 0
                else:
                    if lnk[1] not in ff.extension:
                        ff.extension[lnk[1]] = 0
                extension = lnk[3]
                # Extension specified?
                if extension:
                    # Compute total flow/fund directly imputed to the specific FF
                    t_lst = reversed(lnk[4]) if source else lnk[4]
                    for i, al in enumerate(t_lst):
                        if i == 0:
                            if al in visited_atomic_links:
                                raise Exception("This link should not appear more than one time")
                            if al.extension != extension:
                                raise Exception("Extensions are not coincident")
                        else:
                            if al in visited_atomic_links:
                                al.extension += extension
                            else:
                                visited_atomic_links.add(al)
                                al.extension = extension
                    if source:
                        ff.extension[lnk[0]] += extension
                    else:
                        ff.extension[lnk[1]] += extension

        # TODO Aggregate flow/fund from children FF's (for a hierarchy of FlowFunds)

# ---------------------------------------------------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------------------------------------------------


def get_html_template():
    return """
<!DOCTYPE html>
  <html style="height: 100%;">
    <head>
    <!-- Include the GoJS library. -->
<script src="http://gojs.net/latest/release/go.js"></script>
<script>
  function SankeyLayout() {
    go.LayeredDigraphLayout.call(this);
  }
  go.Diagram.inherit(SankeyLayout, go.LayeredDigraphLayout);

  // Before creating the LayeredDigraphNetwork of vertexes and edges,
  // determine the desired height of each node (Shape).
  /** @override */
  SankeyLayout.prototype.createNetwork = function() {
    this.diagram.nodes.each(function(node) {
      var height = getAutoHeightForNode(node);
      var font = "bold " + Math.max(12, Math.round(height / 8)) + "pt Segoe UI, sans-serif"
      var shape = node.findObject("SHAPE");
      var text = node.findObject("TEXT");
      var ltext = node.findObject("LTEXT");
      if (shape) shape.height = height;
      if (text) text.font = font;
      if (ltext) ltext.font = font;
    });
    return go.LayeredDigraphLayout.prototype.createNetwork.call(this);
  };

  function getAutoHeightForNode(node) {
    var heightIn = 0;
    var it = node.findLinksInto()
    while (it.next()) {
      var link = it.value;
      heightIn += link.computeThickness();
    }
    var heightOut = 0;
    var it = node.findLinksOutOf()
    while (it.next()) {
      var link = it.value;
      heightOut += link.computeThickness();
    }
    var h = Math.max(heightIn, heightOut);
    if (h < 10) h = 10;
    return h;
  };

  // treat dummy vertexes as having the thickness of the link that they are in
  /** @override */
  SankeyLayout.prototype.nodeMinColumnSpace = function(v, topleft) {
    if (v.node === null) {
      if (v.edgesCount >= 1) {
        var max = 1;
        var it = v.edges;
        while (it.next()) {
          var edge = it.value;
          if (edge.link != null) {
            var t = edge.link.computeThickness();
            if (t > max) max = t;
            break;
          }
        }
        return Math.ceil(max/this.columnSpacing);
      }
      return 1;
    }
    return go.LayeredDigraphLayout.prototype.nodeMinColumnSpace.call(this, v, topleft);
  }

  /** @override */
  SankeyLayout.prototype.assignLayers = function() {
    go.LayeredDigraphLayout.prototype.assignLayers.call(this);
    var maxlayer = this.maxLayer;
    // now make sure every vertex with no outputs is maxlayer
    for (var it = this.network.vertexes.iterator; it.next() ;) {
      var v = it.value;
      var node = v.node;
      var key = node.key;
      if (v.destinationVertexes.count == 0) {
        v.layer = 0;
      }
      if (v.sourceVertexes.count == 0) {
        v.layer = maxlayer;
      }
    }
    // from now on, the LayeredDigraphLayout will think that the Node is bigger than it really is
    // (other than the ones that are the widest or tallest in their respective layer).
  };

  /** @override */
  SankeyLayout.prototype.commitLayout = function() {
    go.LayeredDigraphLayout.prototype.commitLayout.call(this);
    for (var it = this.network.edges.iterator; it.next();) {
      var link = it.value.link;
      if (link && link.curve === go.Link.Bezier) {
        // depend on Link.adjusting === go.Link.End to fix up the end points of the links
        // without losing the intermediate points of the route as determined by LayeredDigraphLayout
        link.invalidateRoute();
      }
    }
  };
  // end of SankeyLayout

function init() {
    var GO = go.GraphObject.make;  // for conciseness in defining templates

    function nodeInfo(d) {  // Tooltip info for a node data object
      var str = "Node " + d.key + ": " + d.text + "0x0a";
      if (d.group)
        str += "member of " + d.group;
      else
        str += "top-level node";
      return str;
    }

    function linkInfo(d) {  // Tooltip info for a link data object
      return "Link:0x0afrom " + d.from + " to " + d.to;
    }

    function groupInfo(adornment) {  // takes the tooltip or context menu, not a group node data object
      var g = adornment.adornedPart;  // get the Group that the tooltip adorns
      var mems = g.memberParts.count;
      var links = 0;
      g.memberParts.each(function(part) {
        if (part instanceof go.Link) links++;
      });
      return "Group " + g.data.key + ": " + g.data.text + "0x0a" + mems + " members including " + links + " links";
    }

if ({{sankey}})
{
    myDiagram =
      GO(go.Diagram, "myDiagramDiv", // the ID of the DIV HTML element
        {
          initialAutoScale: go.Diagram.UniformToFill,
          "animationManager.isEnabled": false,
          layout: GO(SankeyLayout,
                    {
                      setsPortSpots: false,  // to allow the "Side" spots on the nodes to take effect
                      direction: 0,  // rightwards
                      layeringOption: go.LayeredDigraphLayout.LayerOptimalLinkLength,
                      packOption: go.LayeredDigraphLayout.PackStraighten || go.LayeredDigraphLayout.PackMedian,
                      layerSpacing: 150,  // lots of space between layers, for nicer thick links
                      columnSpacing: 1
                    })
        });

    var colors = ["#AC193D/#BF1E4B", "#2672EC/#2E8DEF", "#8C0095/#A700AE", "#5133AB/#643EBF", "#008299/#00A0B1", "#D24726/#DC572E", "#008A00/#00A600", "#094AB2/#0A5BC4"];

    // this function provides a common style for the TextBlocks
    function textStyle() {
      return { font: "bold 12pt Segoe UI, sans-serif", stroke: "black", margin: 5 };
    }

    // define the Node template
    myDiagram.nodeTemplate =
      GO(go.Node, go.Panel.Horizontal,
        {
          locationObjectName: "SHAPE",
          locationSpot: go.Spot.MiddleLeft,
          portSpreading: go.Node.SpreadingPacked  // rather than the default go.Node.SpreadingEvenly
        },
        GO(go.TextBlock, textStyle(),
          { name: "LTEXT" },
          new go.Binding("text", "ltext")),
        GO(go.Shape,
          {
            name: "SHAPE",
            figure: "Rectangle",
            fill: "#2E8DEF",  // default fill color
            stroke: null,
            strokeWidth: 0,
            portId: "",
            fromSpot: go.Spot.RightSide,
            toSpot: go.Spot.LeftSide,
            height: 50,
            width: 20
          },
          new go.Binding("fill", "color")),
        GO(go.TextBlock, textStyle(),
          { name: "TEXT" },
          new go.Binding("text"))
      );

    function getAutoLinkColor(data) {
      var nodedata = myDiagram.model.findNodeDataForKey(data.from);
      var hex = nodedata.color;
      if (hex.charAt(0) == '#') {
        var rgb = parseInt(hex.substr(1, 6), 16);
        var r = rgb >> 16;
        var g = rgb >> 8 & 0xFF;
        var b = rgb & 0xFF;
        var alpha = 0.4;
        if (data.width <= 2) alpha = 1;
        var rgba = "rgba(" + r + "," + g + "," + b + ", " + alpha + ")";
        return rgba;
      }
      return "rgba(173, 173, 173, 0.25)";
    }

    // define the Link template
    var linkSelectionAdornmentTemplate =
      GO(go.Adornment, "Link",
        GO(go.Shape,
          { isPanelMain: true, fill: null, stroke: "rgba(0, 0, 255, 0.3)", strokeWidth: 0 })  // use selection object's strokeWidth
      );

    myDiagram.linkTemplate =
      GO(go.Link, go.Link.Bezier,
        {
          selectionAdornmentTemplate: linkSelectionAdornmentTemplate,
          layerName: "Background",
          fromEndSegmentLength: 150, toEndSegmentLength: 150,
          adjusting: go.Link.End
        },
        GO(go.Shape, { strokeWidth: 4, stroke: "rgba(173, 173, 173, 0.25)" },
         new go.Binding("stroke", "", getAutoLinkColor),
         new go.Binding("strokeWidth", "width"))
      );
}
else
{
    if ("{{layout}}"=="ForceDirectedLayout")
    {
    myDiagram = GO(go.Diagram, "myDiagramDiv",  // create a Diagram for the DIV HTML element
                  {
                    initialContentAlignment: go.Spot.Center,  // center the content
                    initialAutoScale: go.Diagram.UniformToFill,
                    layout: GO(go.ForceDirectedLayout), // TreeLayout, ForceDirectedLayout, LayeredDigraphLayout
                    "undoManager.isEnabled": true  // enable undo & redo
                  });
    }
    else
    {
    myDiagram =
      GO(go.Diagram, "myDiagramDiv", // the ID of the DIV HTML element
        {
          initialAutoScale: go.Diagram.UniformToFill,
          "animationManager.isEnabled": false,
          layout: GO(go.LayeredDigraphLayout,
                    {
                      setsPortSpots: false,  // to allow the "Side" spots on the nodes to take effect
                      direction: 0,  // rightwards
                      layeringOption: go.LayeredDigraphLayout.LayerOptimalLinkLength,
                      packOption: go.LayeredDigraphLayout.PackStraighten || go.LayeredDigraphLayout.PackMedian,
                      layerSpacing: 150,  // lots of space between layers, for nicer thick links
                      columnSpacing: 1
                    })
        });
    }
    // define a simple Node template
    myDiagram.nodeTemplate =
      GO(go.Node, "Auto",  // the Shape will go around the TextBlock
        GO(go.Shape, "RoundedRectangle", { strokeWidth: 0},
          // Shape.fill is bound to Node.data.color
          new go.Binding("fill", "color")),
        GO(go.TextBlock,
          {
            font: "14px sans-serif",
            stroke: '#333',
            margin: 6,  // make some extra space for the shape around the text
            isMultiline: false,  // don't allow newlines in text
            editable: true  // allow in-place editing by user
          },
          new go.Binding("text", "text").makeTwoWay()),  // the label shows the node data's text
        { // this tooltip Adornment is shared by all nodes
          toolTip:
            GO(go.Adornment, "Auto",
              GO(go.Shape, { fill: "#FFFFCC" }),
              GO(go.TextBlock, { margin: 4 },  // the tooltip shows the result of calling nodeInfo(data)
                new go.Binding("text", "", nodeInfo))
            )
        }
      );

    // The link shape and arrowhead have their stroke brush data bound to the "color" property
    myDiagram.linkTemplate =
      GO(go.Link,
         {corner: 10, curve: go.Link.JumpOver},
         go.Link.Bezier,
         { toShortLength: 3, relinkableFrom: true, relinkableTo: true },  // allow the user to relink existing links
        GO(go.Shape,
          { strokeWidth: 2 },
          new go.Binding("stroke", "color")),
        GO(go.Shape,
          { toArrow: "Standard", stroke: null },
          new go.Binding("fill", "color")),
        { // this tooltip Adornment is shared by all links
          toolTip:
            GO(go.Adornment, "Auto",
              GO(go.Shape, { fill: "#FFFFCC" }),
              GO(go.TextBlock, { margin: 4 },  // the tooltip shows the result of calling linkInfo(data)
                new go.Binding("text", "", linkInfo))
            ),
        }
      );

    // Groups consist of a title in the color given by the group node data
    // above a translucent gray rectangle surrounding the member parts
    myDiagram.groupTemplate =
      GO(go.Group, "Vertical",
        { selectionObjectName: "PANEL",  // selection handle goes around shape, not label
          ungroupable: true },  // enable Ctrl-Shift-G to ungroup a selected Group
        GO(go.TextBlock,
          {
            font: "bold 19px sans-serif",
            isMultiline: false,  // don't allow newlines in text
            editable: true  // allow in-place editing by user
          },
          new go.Binding("text", "text").makeTwoWay(),
          new go.Binding("stroke", "color")),
        GO(go.TextBlock,
          {
            font: "bold 14px sans-serif",
            stroke: '#333',
            margin: 6,  // make some extra space for the shape around the text
            isMultiline: false,  // don't allow newlines in text
            editable: true  // allow in-place editing by user
          },
          new go.Binding("text", "text").makeTwoWay()),  // the label shows the node data's text
        GO(go.Panel, "Auto",
          { name: "PANEL" },
          GO(go.Shape, "Rectangle",  // the rectangular shape around the members
            { fill: "rgba(128,128,128,0.2)", stroke: "gray", strokeWidth: 3 }),
          GO(go.Placeholder, { padding: 10 })  // represents where the members are
        ),
        { // this tooltip Adornment is shared by all groups
          toolTip:
            GO(go.Adornment, "Auto",
              GO(go.Shape, { fill: "#FFFFCC" }),
              GO(go.TextBlock, { margin: 4 },
                // bind to tooltip, not to Group.data, to allow access to Group properties
                new go.Binding("text", "", groupInfo).ofObject())
            ),
        }
      );
}

    myDiagram.model = new go.GraphLinksModel(
        {{nodes}},
        {{links}}
    );
}

</script>
    <body onload="init()" style="height: 100%;">
        <!-- The DIV for a Diagram needs an explicit size or else we won't see anything.
             In this case we also add a border to help see the edges. -->
        <div id="myDiagramDiv" style="border: solid 1px blue; width:%99; height:99%"></div>
     </body>
  </html>
    """

colors = ["lightgreen", "coral", "chartreuse", "aquamarine",  "cornflowerblue", "cyan", "darkcyan", "blueviolet",
          "chocolate", "darkgreen", "darkorange", "darkkhaki", "darksalmon", "darkslategray", "darkturquoise",
          "deeppink", "firebrick", "forestgreen", "fuchsia", "gold", "greenyellow", "indianred", "indigo", "lawngreen"
          "lightseagreen", "lime", "magenta", "maroon", "mediumaquamarine", "mediumpurple",
          "mediumseagreen", "midnightblue", "olive", "orange", "lightblue"]


def generate_graph_one(case_study):
    """
    A graph with ALL the elements represented
    :param case_study: The case study to represent
    :return: A string with the full HTML page, ready to be painted
    """

    def get_nodes(p, p_id):
        d_p[p_id] = p
        c_id = p_id
        for c in p.children:
            links.append((p_id+1, c_id))
            p_id = get_nodes(c, p_id+1)
        return p_id

    # ---------------------------------

    # Get nodes: the Processors. key, text, color, group, isGroup
    d_p = {}  # Processors map
    links = []  # Array of links

    cont = 1
    for root in case_study.root_processors:
        cont = get_nodes(root, cont)

    # Convert links
    n_lst = ['{"key": "'+str(k)+'", "text": "'+d_p[k].name+'", "color": "lightblue"}' for k in d_p]
    l_lst = ['{"from": "'+str(t[0])+'", "to": "'+str(t[1])+'"}' for t in links]
    nodes = "["+",\n".join(n_lst)+"]"
    links = "["+",\n".join(l_lst)+"]"
    # Get links: hierarchical relation. from, to, color
    return get_html_template().replace("{{nodes}}", nodes).replace("{{links}}", links).replace("{{sankey}}", "false")


def generate_graph_two(case_study, type="structure"):
    """
    A graph with ALL the elements represented
    :param case_study: The case study to represent
    :param type: "structure", "sankey"
    :return: A string with the full HTML page, ready to be painted. Also, a name for the type of graph (for file name generation)
    """
    def get_nodes_1(p, p_id):
        d_p[p_id] = p
        p_d[p] = p_id
        c_id = p_id
        for c in p.children:
            links.append((p_id+1, c_id))
            p_id = get_nodes_1(c, p_id+1)
        return p_id

    def get_nodes_2(ff, f_id):
        d_p[f_id] = ff
        p_d[ff] = f_id
        c_id = f_id
        for c in ff.children:
            links.append((f_id+1, c_id))
            f_id = get_nodes_2(c, f_id+1)
        return f_id

    def get_flow_links(ff, c_idx):
        # Links related directly to the flow
        if ff in case_study.high_level_links:
            tmp = set()
            for t in case_study.high_level_links[ff]:
                if t[0] not in tmp:
                    # A link from the FlowFund to a "source" or sink Processor
                    source = not ff.is_waste_flow
                    if source:
                        f_links.append((p_d[ff], p_d[t[0]], colors[c_idx], ff.extension[t[0]]))
                    else:
                        f_links.append((p_d[t[1]], p_d[ff], colors[c_idx], ff.extension[t[1]]))
                    tmp.add(t[0])
            f_links.extend([(p_d[t.source.processor], p_d[t.destination.processor], colors[c_idx], t.extension) for t in ff.links])
            c_idx += 1
        for c in ff.children:
            c_idx = get_flow_links(c, c_idx)

        return c_idx

        # For each Flow, find a color. Then find all its appearances

    # ---------------------------------

    # Get nodes: the Processors. key, text, color, group, isGroup
    d_p = {}  # Processors and FlowsFunds map
    p_d = {}  # Reverse node map
    links = []  # Array of hierarchical links
    f_links = []  # Array of flow links

    cont = 1
    for root in case_study.root_processors:
        cont = get_nodes_1(root, cont)

    last_proc = cont

    for root in case_study.root_flowsfunds:
        cont = get_nodes_2(root, cont+1)

    # Compute bottom-up extensions
    case_study.compute_bottom_up_flowfunds()

    # Links for flows
    c_idx = 0
    for root in case_study.root_flowsfunds:
        c_idx = get_flow_links(root, c_idx)

    # Convert nodes and hierarchical links to strings
    n_lst = ['{"key": "'+str(k)+'", "text": "'+d_p[k].name+'", "color": "lightblue"}' for k in d_p if k <= last_proc]
    ff_lst = ['{"key": "'+str(k)+'", "text": "'+d_p[k].name+'", "color": "'+("orange" if d_p[k].roegen_type==FlowFundRoegenType.flow else "gold")+'"}' for k in d_p if k > last_proc]
    color = "#5c6567"  # Gold
    if type == "structure":
        l_lst = ['{"from": "'+str(t[0])+'", "to": "'+str(t[1])+'", "color": "'+color+'"}' for t in links]
    else:
        l_lst = []
    fl_lst = ['{"from": "'+str(t[0])+'", "to": "'+str(t[1])+'", "color": "'+t[2]+'", "width": "'+str(t[3])+'"}' for t in f_links]
    if ff_lst:
        n_lst.extend(ff_lst)
    if fl_lst:
        l_lst.extend(fl_lst)
    nodes = "["+",\n".join(n_lst)+"]"
    links = "["+",\n".join(l_lst)+"]"
    # Get links: hierarchical relation. from, to, color
    html = get_html_template().replace("{{nodes}}", nodes).replace("{{links}}", links).\
        replace("{{sankey}}", "false" if "structure" in type else "true").\
        replace("{{layout}}", "ForceDirectedLayout" if type == "structure" else "LayeredDigraphLayout" if type == "structure2" else "LayeredDigraphLayout")

    if type == "structure":
        name = "graph_all_elements"
    elif type=="sankey":
        name = "sankey"

    return html, case.name + "_" + name

# ---------------------------------------------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------------------------------------------


def cs_test():
    # Hierarchy of Processors
    #    eco
    #     |
    #     p1
    #    /  \
    #   p2  p3
    #  /  \
    # p4   p5
    eco = Processor(None, "Environment", ProcessorType.environment)  # Root Processor (Earth!)
    p1 = Processor(eco, "Society")
    p2 = Processor(p1, "Productive Sectors")
    p3 = Processor(p1, "Household")
    p4 = Processor(p2, "Primary")
    p5 = Processor(p2, "Secondary & Tertiary")

    # Fund tree with a single node
    ha = FlowFund(None, "Hours Available", FlowFundRoegenType.fund)
    # Another trivial Fund tree
    lu = FlowFund(None, "Land Use", FlowFundRoegenType.fund)
    # Waste
    co2 = FlowFund(None, "CO2", FlowFundRoegenType.flow, True)

    # "The Case under Study"
    case = CaseStudy("test1", [eco], [ha, lu, co2])

    # case.p(["ECO", ["S", ["PS", "HH"]]])
    # Connect ECO Processor to "Productive Sectors" with extension of 3, passing -automatically- through "Society"
    case.link(p3, p4, ha, 3)
    case.link(p3, p5, ha, 6)
    # Connect ECO Processor to "Household" with extension of 3, passing -automatically- through "Society"
    # case.link(eco, p3, ha, 5)

    if False:
        # Repeat the connection pattern of HA for LU (connects ECO to PS and HH)
        # (still no way to specify "extension" of the flows)
        case.clone_links_pattern(ha, lu)
    else:
        case.link(eco, p4, lu, 10)
        case.link(eco, p5, lu, 8)
        # Connect ECO Processor to "Household" with extension of 3, passing -automatically- through "Society"
        case.link(eco, p3, lu, 15)

    case.link(p5, eco, co2, 20)

    return case


def cs_lost_one():
    """
    Case Study based on Michele Staiano's "Lost" Case Study, focused on food production for a population of 100 persons
    in ideal conditions (unlimited energy, unlimited soil, unlimited chemicals provision)
    :return:
    """
    eco = Processor(None, "Environment", ProcessorType.environment)  # Root Processor (Earth!)
    p1 = Processor(eco, "Society")
    p2 = Processor(p1, "Agriculture")
    p3 = Processor(p2, "Low intens Agr")
    p4 = Processor(p2, "Mid intens Agr")
    p5 = Processor(p2, "High intens Agr")

    # Fund tree with a single node
    ha = FlowFund(None, "Human Time", FlowFundRoegenType.fund)
    # Another trivial Fund tree
    lu = FlowFund(None, "Land Use", FlowFundRoegenType.fund)
    # Food (grain equivalent)
    gr = FlowFund(None, "Food", FlowFundRoegenType.flow)

    # "The Case under Study"
    case = CaseStudy("LOST_1", [eco], [ha, lu, gr])

    # Four different grain consumption scenarios considered
    #
    # Veg:      21313
    # Mod.meat: 29667
    # Big meat: 37169
    # Mix:      30161 (this is just another scenario)
    #

    # Links will not change for scenario to scenario, only their extensions
    # Land Use
    case.link(eco, p3, lu)
    case.link(eco, p4, lu)
    case.link(eco, p5, lu)
    # Food
    case.link(p3, p1, gr)
    case.link(p4, p1, gr)
    case.link(p5, p1, gr)
    # Human labor for agriculture
    case.link(p1, p3, ha)
    case.link(p1, p4, ha)
    case.link(p1, p5, ha)

    return case


def cs_fuel_supply():
    eco = Processor(None, "Environment", ProcessorType.environment)  # Root Processor (Earth!)
    p1 = Processor(eco, "Society")
    p2 = Processor(p1, "Productive Sectors")
    p3 = Processor(p1, "Household")
    p4 = Processor(p2, "Primary")
    p5 = Processor(p2, "Secondary & Tertiary")
    p6 = Processor(p5, "Energy Sector")
    p7 = Processor(p6, "Fuel Supply")
    p8 = Processor(p7, "Extraction")
    p9 = Processor(p7, "Transport #1")
    p10 = Processor(p7, "Refinement")
    p11 = Processor(p7, "Transport #2")

    p12 = Processor(p8, "On-shore")
    p13 = Processor(p8, "Off-shore")
    p14 = Processor(p9, "Ships")
    p15 = Processor(p9, "Pipelines")
    p16 = Processor(p9, "Trucks")
    p17 = Processor(p10, "Small")
    p18 = Processor(p10, "Medium")
    p19 = Processor(p10, "Large")
    p20 = Processor(p11, "Ships")
    p21 = Processor(p11, "Trucks")
    p22 = Processor(p11, "Pipelines")

    oil = FlowFund(None, "Crude Oil", FlowFundRoegenType.fund)

    case = CaseStudy("Fuel Supply 1", [eco], [oil])

    # case.link(eco, p8, oil)
    # case.link(p8, p12, oil)
    # case.link(p8, p13, oil)

    # TODO Arrange LINKS adequately
    # case.link(p8)

    return case


def cs_soslaires():
    eco = Processor(None, "Environment", ProcessorType.environment)  # Root Processor (Earth!)
    p0 = Processor(eco, "Other Economy")
    p1_ = Processor(eco, "Society")
    p1__ = Processor(p1_, "Household")
    p1 = Processor(p1_, "Soslaires")
    p2 = Processor(p1, "Wind Farm")
    p3 = Processor(p1, "Desalination Plant")
    p4 = Processor(p1, "Farm")
    p5 = Processor(p4, "Crop #1")
    p6 = Processor(p4, "Crop #2")
    p7 = Processor(p4, "Crop #3")

    # ------------------------------------

    ha = FlowFund(None, "Human Time", FlowFundRoegenType.fund)
    lu = FlowFund(None, "Land Use", FlowFundRoegenType.fund)
    wpc = FlowFund(None, "Wind Power Capacity", FlowFundRoegenType.fund)
    dpc = FlowFund(None, "Desalination Power Capacity", FlowFundRoegenType.fund)
    ic = FlowFund(None, "Irrigation Capacity", FlowFundRoegenType.fund)

    wd = FlowFund(None, "Wind", FlowFundRoegenType.flow)
    sw = FlowFund(None, "Sea Water", FlowFundRoegenType.flow)

    wb = FlowFund(None, "Brine", FlowFundRoegenType.flow)
    wo = FlowFund(None, "Organic Waste", FlowFundRoegenType.flow)
    wpol = FlowFund(None, "Difussive Polution", FlowFundRoegenType.flow)
    wco2 = FlowFund(None, "CO2", FlowFundRoegenType.flow)

    ach = FlowFund(None, "Agrochemicals", FlowFundRoegenType.flow)
    fuel = FlowFund(None, "Fuel", FlowFundRoegenType.flow)

    veg = FlowFund(None, "Vegetables", FlowFundRoegenType.flow)

    # -------------------------------------
    case = CaseStudy("Soslaires 1", [eco], [ha, lu, wpc, dpc, ic, wd, sw, wb, wo, wpol, wco2, ach, fuel, veg])

    case.link(eco, p2, wd)
    case.link(eco, p3, sw)

    case.link(p3, eco, wb)
    case.link(p4, eco, wo)
    case.link(p4, eco, wpol)
    case.link(p4, eco, wco2)

    case.link(eco, p2, lu)
    case.link(eco, p3, lu)
    case.link(eco, p4, lu)

    case.link(p1__, p2, ha)
    case.link(p1__, p3, ha)
    case.link(p1__, p4, ha)

    case.link(p1_, p2, wpc)
    case.link(p1_, p3, dpc)
    case.link(p1_, p4, ic)

    case.link(p0, p4, ach)
    case.link(p0, p4, fuel)
    case.link(p4, p1__, veg)

    return case


if __name__ == '__main__':
    # case = cs_lost_one()
    # case = cs_fuel_supply()
    case = cs_soslaires()
    case = cs_test()
    case = cs_fuel_supply()

    s, name = generate_graph_two(case, "structure")  # "structure", "structure2", "sankey"
    with open("/home/rnebot/"+name+".html", "wt") as f:
        f.write(s)


