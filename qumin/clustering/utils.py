# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Utilities used in clustering.

Author:Sacha Beniamine.
"""
try:
    from matplotlib import pyplot

    plt = pyplot
    matplotlib_loaded = True
except:
    matplotlib_loaded = False

try:
    import networkx as nx

    nx_loaded = True
except:
    nx_loaded = False

import numpy as np
import re
import logging

log = logging.getLogger()
from collections import defaultdict


class Node(object):
    """Represent an inflection class tree.

    Attributes:
        labels (list): labels of all the leaves under this node.
        children (list): direct children of this node.
        attributes (dict): attributes for this node.
            Currently, three attributes are expected:
            size (int): size of the group represented by this node.
            DL (float): Description length for this node.
            color (str): color of the splines from this node to its children, in a format usable by pyplot.
            Currently, red ("r") is used when the node didn't decrease Description length, blue ("b") otherwise.
            macroclass (bool): Is the node in a macroclass ?
            macroclass_root (bool): Is the node the root of a macroclass ?

            The attributes "_x" and "_rank" are reserved,
            and will be overwritten by the draw function.
    """

    def __init__(self, labels, children=None, **kwargs):
        """Node constructor.

        Arguments:
            labels (iterable): labels of all the leaves under this node.
            children (list): direct children of this node.
            kwargs: any other keyword argument will be added as node attributes.
             Note that certain algorithm expect the Node to have (int) "size",
             (str) "color", (bool) "macroclass", or (float) "DL" attributes.

            The attributes "_x" and "_rank" are reserved,
            and will be overwritten by the draw function.
        """
        self.labels = sorted(labels)
        self.children = children if children else []
        self.attributes = kwargs
        self.istree = self._test_if_tree()

    def _test_if_tree(self):
        """ Test if this is a tree by checking in-degree."""
        parents = {}
        for node in self:
            for child in node.children:
                if child in parents:
                    return False
                parents[child] = node
        return True

    def __str__(self):
        """Return a repr string for Nodes."""
        attrs = " - ".join(
            "{}={}".format(key, self.attributes[key]) for key in self.attributes)
        return "< Node object - " + ", ".join(self.labels) + " - " + attrs + ">"

    def macroclasses(self, parent_is_macroclass=False):
        """Find all the macroclasses nodes in this tree"""
        self_is_macroclass = self.attributes["macroclass"]
        if not parent_is_macroclass and self_is_macroclass:
            labels = self.labels
            return {labels[0]: labels}
        elif self.children:
            macroclasses_under = {}
            for child in self.children:
                child_macroclasses = child.macroclasses(
                    parent_is_macroclass=self_is_macroclass)
                macroclasses_under.update(child_macroclasses)
            return macroclasses_under
        return {}

    def _recursive_xy(self, ticks, node_spacing):
        if self.attributes.get("_y", None) is None:
            half_step = node_spacing // 2
            y = 1
            if len(self.children) > 0:
                xs, ys = zip(
                    *[child._recursive_xy(ticks, node_spacing) for child in self.children])
                y += max(ys)
                xs = sorted(xs)
                center = xs[0] + ((xs[-1] - xs[0])/2)
                if y in ticks:

                    # If the preferred value is far enough, pick it
                    dist_mean_x = min(abs(center-x2) for x2 in ticks[y])
                    if dist_mean_x >= half_step:
                        x = center
                    else:
                        # Otherwise, candidates are all ticks in the node's span
                        candidates = np.arange(xs[0]-half_step, xs[-1] + half_step, half_step).tolist()

                        # Pick the candidate which is the further from existing points
                        x = max(candidates,
                                key=lambda x1: min(abs(x1 - x2) for x2 in ticks[y]))

                    ticks[y].append(x)
                else:
                    x = center
                    ticks[y] = [x]
                self.attributes["_x"] = x
            self.attributes["_y"] = y
        return self.attributes["_x"], self.attributes["_y"]

    def _erase_xy(self):
        if "_x" in self.attributes:
            del self.attributes["_x"]
        if "_y" in self.attributes:
            del self.attributes["_y"]
        for child in self.children:
            child._erase_xy()

    def _compute_xy(self, layout="qumin", pos=None):
        graphviz_layout = nx.drawing.nx_agraph.graphviz_layout
        nx_layouts = {"dot": lambda x: graphviz_layout(x, prog="dot"),
                      # "spring":nx.drawing.spring_layout,
                      # "kamada_kawai":nx.drawing.kamada_kawai_layout,
                      #   "radial": lambda x: graphviz_layout(x, prog="twopi"),
                      }
        if "_y" in self.attributes and self.attributes["_y"] is not None:
            self._erase_xy()

        if layout == "qumin":  # For trees
            leaves_ordered = self._sort_leaves()
            x = 0
            step = 30
            for leaf in leaves_ordered:
                leaf.attributes["_x"] = x
                x += step
            self._recursive_xy({}, step)
        elif layout in nx_layouts:
            infimum = Node(["infimum"])
            layout_f = nx_layouts[layout]

            # add infimum to improve node placement
            leaves = list(self.leaves())
            for i, leaf in enumerate(leaves):
                leaf.children.append(infimum)

            if pos is None:
                pos = layout_f(self.to_networkx())

            for leaf in leaves:
                leaf.children = []

            for node in self:
                # annotate x and y
                node.attributes["_x"], node.attributes["_y"] = pos[tuple(node.labels)]
        else:
            raise NotImplementedError("Layout method {} is not implemented."
                                      "pick from: qumin, {}".format(layout,
                                                                    ", ".join(
                                                                        nx_layouts)))

    def _sort_leaves(self):
        """Sorts leaves by similarity for plotting"""
        leaves = list(self.leaves())
        li = len(leaves)
        similarities = np.zeros((li, li))
        ancestors = defaultdict(set)
        for node in self:
            for l in node.labels:
                ancestors[(l,)].add(tuple(node.labels))
            # for c in node.children:
            #     if len(c.labels) == 1:
            #         ancestors[tuple(c.labels)].add(tuple(node.labels))

        for i, leaf in enumerate(leaves):
            for j, leaf2 in enumerate(leaves):
                if i != j:
                    a1 = ancestors[tuple(leaf.labels)]
                    a2 = ancestors[tuple(leaf2.labels)]
                    jaccard = len(a1 & a2) / len(a1 | a2)
                    similarities[i, j] = similarities[j, i] = jaccard
        i, j = np.unravel_index(np.argmax(similarities, axis=None),
                                similarities.shape)
        leaves_ordered = [leaves[i], leaves[j]]
        similarities[:, i] = float("-inf")
        similarities[:, j] = float("-inf")
        # mark visited
        for _ in range(li - 2):
            k1 = np.argmax(similarities[i, :])
            k2 = np.argmax(similarities[j, :])
            if k1 > k2:
                leaves_ordered.insert(0,leaves[k1])
                similarities[:, k1] = float("-inf")  # mark visited
                i = k1
            else:
                leaves_ordered.append(leaves[k2])
                similarities[:, k2] = float("-inf")  # mark visited
                j = k2
        return leaves_ordered

    def to_networkx(self):
        if not nx_loaded:
            raise ImportError("Can't convert to networkx, because it couldn't be loaded.")
        G = nx.DiGraph()
        for node in self:
            name = tuple(node.labels)
            G.add_node(name, **node.attributes)
            for child in node.children:
                childname = tuple(child.labels)
                G.add_edge(name, childname)
        return G

    def __eq__(self, other):
        return (self.labels == other.labels) & (set(self.children) == set(other.children))

    def __repr__(self):
        rules = [
            str(node.labels) + " -> " + " ".join(str(c.labels) for c in sorted(node.children, key=lambda x:x.labels)) for
            node in self]
        return "\n".join(sorted(rules))

    def __hash__(self):
        return hash(repr(self))

    def __iter__(self):
        agenda = [self]
        nodes = []
        while agenda:
            node = agenda.pop(0)
            if not node.attributes.get("visited", False):
                node.attributes["visited"] = True
                agenda = node.children + agenda
                nodes.append(node)

        # cleanup
        for n in nodes:
            del n.attributes["visited"]

        yield from nodes

    def leaves(self):
        return filter(lambda x: not bool(x.children), self)

    def tree_string(self):
        """Return the inflection class tree as a string with parenthesis.

        Assumes size, DL and color attributes,
        with color = "r" if this is above a macroclass.

        Example:
            In the label, fields are separated by "#" as such::

                (<labels>#<size>#<DL>#<color> )
        """
        if not self.istree:
            raise NotImplementedError(
                "Tree string is only possible with trees, this looks like a graph.")
        labels = "&".join(self.labels)
        ignore = ["_rank", "_x", "_y", "macroclass", "macroclass_root", "point_settings"]
        attributes = "#".join(
            "{}={}".format(key, str(self.attributes[key]).replace(" ", "_")) for key in
            sorted(self.attributes) if
            key not in ignore)
        children_str = [repr(child) for child in self.children]
        string = "(" + labels + "#" + attributes + " ".join([""] + children_str) + " )"

        return string

    def to_tikz(self, leavesfunc=None, nodefunc=None, layout="qumin", pos=None,
                ratio=(1, 1), width=20, color_attrs=None):
        color_attrs = color_attrs or []

        def attribute_table(node):
            table = []
            common = len(node.attributes.get("common", [])) > 0
            if not node.children:
                table.append(node.labels[0])
                if common:
                    table.append("\\\\")
            if common:
                table.append("\\begin{tabular}{rl}")
                for feature in node.attributes["common"]:
                    a, b = feature.split("=")
                    table.append("\\textsc{" + a + "} & " + b + " \\\\")
                table.append("\end{tabular}")
            return "".join(table)

        def scale(m, max_obs, max_target):
            return round((m / max_obs) * max_target, 2)

        def get_anchor(node):
            if node.children:
                return ("north", "south")
            return ("south", "north")

        nodefunc = nodefunc or attribute_table
        leavesfunc = leavesfunc or attribute_table
        self._compute_xy(pos=pos, layout=layout)

        node_template = "\\node ({name}) at ({x},{y}) [shape=circle,draw=black,inner sep=0," \
                        "fill=black,minimum size=3pt] {{}};"
        edge_template = "\\draw ({a}.{anchor_a}) edge[-] ({b}.{anchor_b});"
        node_annot = "\\node[anchor={anchor_self},align=left{style}] at ({a}.{anchor_pt}) {{{label}}};"

        xys = list(zip(*[(n.attributes["_x"], n.attributes["_y"]) for n in self]))
        max_x = max(xys[0])
        max_y = max(xys[1])
        height = width * (ratio[1] / ratio[0])

        lines = []

        for i, node in enumerate(self):
            label_f = nodefunc if node.children else leavesfunc
            style = ""
            if not node.children:
                style = ",draw,fill=white,fill opacity=.8"
            elif i == 0 or not node.attributes.get("common", False):
                style = ",draw=none,fill=none"

            if color_attrs and \
                    any([f.split("=")[0] in color_attrs \
                         for f in node.attributes.get("common", [])]):
                style = ",fill=gray!20"

            node_point = node_template.format(name=str(i),
                                              x=scale(node.attributes["_x"],
                                                      max_x, width),
                                              y=scale(node.attributes["_y"],
                                                      max_y, height),
                                              )
            a,b = get_anchor(node)
            node_text = node_annot.format(a=str(i),
                                          label=label_f(node),
                                          style=style,
                                          anchor_pt=a,
                                          anchor_self=b,
                                          )
            lines.append(node_point)
            lines.append(node_text)
            node.attributes["tikz-label"] = str(i)

        lines.append("\\begin{pgfonlayer}{background}")
        for node in self:
            a = node.attributes["tikz-label"]
            for child in node.children:
                lines.append(edge_template.format(a=a, b=child.attributes["tikz-label"],
                                                  anchor_a="center",
                                                  anchor_b="center"))

        lines.append("\\end{pgfonlayer}")
        return "\n".join(lines)

    def draw(self, horizontal=False, square=False,
             leavesfunc=lambda n: n.labels[0],
             nodefunc=None, label_rotation=None,
             annotateOnlyMacroclasses=False, point=None,
             edge_attributes=None, interactive=False, layout=False, pos=None, **kwargs):
        """Draw the tree as a dendrogram-style pyplot graph.

        Example::

                                      square=True        square=False

                                     │  ┌──┴──┐         │    ╱╲
            horizontal=False         │  │   ┌─┴─┐       │   ╱  ╲
                                     │  │   │   │       │  ╱   ╱╲
                                     │  │   │   │       │ ╱   ╱  ╲
                                     │__│___│___│       │╱___╱____╲

                                    │─────┐             │⟍
                                    │───┐ ├             │  ⟍
            horizontal=True         │   ├─┘             │⟍ ⟋
                                    │───┘               │⟋
                                    │____________       │____________


        Arguments:
            horizontal (bool):
                Should the tree be drawn with leaves on the y axis ?
                (Defaults to False: leaves on x axis).
            square (bool):
                Should the tree splines be squared with 90° angles ?
                (Defauls to False)
            leavesfunc (fun):
                A function that will be applied to leaves
                before writing them down. Takes a Node, returns a str.
            nodefunc (fun):
                A function that will be applied to nodes
                to annotate them. Takes a Node, returns a str.
            keep_above_macroclass (bool):
                For macroclass history trees: Should the edges above macroclasses be drawn ?
                (Defaults to True).
            annotateOnlyMacroclasses : For macroclass history trees:
                If `True` and nodelabel isn't `None`,
                only the macroclasses nodes are annotated.
            point (fun):
                A function that maps a node to point attributes.
            edge_attributes (fun):
                A function that maps a pair of nodes to edge attributes.
                    By default, use the parent's color and "-" linestyle for nodes,
                    "--" for leaves.
            interactive (bool):
                Whether this is destined to create an interactive plot.
            layout (bool):
                layout keyword, either of "qumin" or "dot". Ignored if pos is given.
            pos (dict):
                A dictionnary of node label to x,y positions.
                Compatible with networkx layout functions.
                If absent, use networkx's graphviz layout.
        """
        self._compute_xy(pos=pos, layout=layout)

        if not matplotlib_loaded:
            return str(self)
        else:
            def annotate(node):
                should_annotate = (not annotateOnlyMacroclasses) or \
                                  node.attributes["macroclass_root"]
                if should_annotate and nodefunc is not None:
                    return str(nodefunc(node))
                return ""

            def default_edge_attributes(node, child):
                attributes = {"linestyle": "-" if node.children else "--",
                              "color": node.attributes.get("color", "#333333")}
                if edge_attributes is not None:
                    attributes.update(edge_attributes(node, child))
                return attributes

            if horizontal:
                textoffset = (5, 0)
                lva = "center"
                lha = "right"
                va = "center"
                ha = "left"
                r = 0

                def coords(node):
                    return node.attributes["_y"], node.attributes["_x"]
            else:
                lva = "top"
                lha = "center"
                va = "bottom"
                ha = "center"
                textoffset = (0, 5)
                r = 45

                def coords(node):
                    return node.attributes["_x"], node.attributes["_y"]
            if label_rotation is not None:
                r = label_rotation

            ax = plt.gca()
            # bg = ax.patch.get_facecolor()

            lines = []
            all_nodes = []

            for node in self:
                this_x, this_y = coords(node)

                # Plot the arcs
                for child in node.children:
                    child_x, child_y = coords(child)
                    attr = default_edge_attributes(node, child)
                    if square:
                        if horizontal:
                            l = ax.plot((this_x, this_x, child_x),
                                        (this_y, child_y, child_y), **attr)
                        else:
                            l = ax.plot((this_x, child_x, child_x),
                                        (this_y, this_y, child_y), **attr)
                    else:
                        l = ax.plot((this_x, child_x), (this_y, child_y), **attr)
                    lines.extend(l)

                # Plot the point
                if point is not None:
                    coll = ax.scatter((this_x,), (this_y,), **point(node))
                    lines.append(coll)

                # Write annotations
                if node.labels:
                    # tmp = node.labels
                    # node.labels = list(microclass)
                    f = nodefunc if node.children else leavesfunc
                    plt.annotate(f(node), xy=(this_x, this_y), xycoords='data', va=lva,
                                 ha=lha, rotation=r)
                    # node.labels = tmp
                else:
                    text = annotate(node)
                    if text is not None:
                        plt.annotate(text, xy=(this_x, this_y), va=va, ha=ha,
                                     textcoords='offset points',
                                     xytext=textoffset)

                all_nodes.append(node)

            # Scale axes
            ax.autoscale()

            if interactive:
                if horizontal:
                    ax.margins(x=0.3, y=0.1)
                else:
                    ax.margins(y=0.3, x=0.1)

            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks
                top='off',  # ticks along the top edge
                bottom='off',  # ticks along the bottom edge
                right='off',  # ticks along the right edge
                left='off',  # ticks along the left edge
                labelbottom='off'
            )
            plt.yticks([], [])
            plt.xticks([], [])
        return lines, all_nodes


def string_to_node(string, legacy_annotation_name=None):
    """Parse an inflection tree written as a string.

    Example:
        In the label, fields are separated by "#" as such::

         (<labels>#<size>#<DL>#<color> (... ) (... ) )

    Returns:
        inflexClass.Node: The root of the tree
    """
    legacy = False
    if "=" not in string:
        legacy = True

    def parse_node(line):
        splitted = line[1:].split("#")
        labels = splitted[0]
        attributes = dict(attr.split("=") for attr in splitted[1:])
        return labels, attributes

    def parse_node_legacy(item):
        item = item[1:].split("#")
        return item

    stack = []
    # plus robuste que de splitter sur l'espace, autorise les espaces dans les lexèmes & attributs
    items = re.split(" (?=[()])", string)

    if legacy_annotation_name:
        annotation_name = legacy_annotation_name
    else:
        annotation_name = "DL"
    for item in items:

        if item[0] == "(":

            if legacy:
                # Backward compatibility mode
                labels, size, annotation, color = parse_node_legacy(item)
                if not color:
                    color = "c"
                if not annotation:
                    annotation = ""
                attributes = {"size": float(size),
                              annotation_name: annotation,
                              "color": color,
                              "macroclass": color == "r"}
            else:
                labels, attributes = parse_node(item)
                if 'color' not in attributes:
                    attributes['color'] = "c"
                attributes['macroclass'] = attributes['color'] != 'r'
            labels = sorted(labels.split("&"))
            attributes['macroclass_root'] = False
            if annotation_name in attributes:
                try:
                    attributes[annotation_name] = float(attributes[annotation_name])
                except ValueError:
                    pass  # only convert numbers

            stack.append(Node(labels, **attributes))

        if item[0] == ")":
            if len(item) > 1:
                log.warning("Warning, bad format ! #{}#".format(item))
            if len(stack) > 1:
                child = stack.pop(-1)
                parent = stack[-1]
                # Macroclass are one level below the red :
                child.attributes['macroclass_root'] = parent.attributes[
                                                          'color'] == 'r' and \
                                                      child.attributes['color'] != 'r'
                stack[-1].children.append(child)
            else:
                return stack[0]

    if len(stack) > 1:
        log.warning("unmatched parenthesis or no root ! " + str(stack))

    log.info(stack[0])
    return stack[0]


def find_microclasses(paradigms):
    """Find microclasses in a paradigm (lines with identical rows).

    This is useful to identify an exemplar of each inflection microclass,
    and limit further computation to the collection of these exemplars.

    Arguments:
        paradigms (pandas.DataFrame):
            a dataframe containing inflectional paradigms.
            Columns are cells, and rows are lemmas.

    Return:
        microclasses (dict).
            classes is a dict. Its keys are exemplars,
            its values are lists of the name of rows identical to the exemplar.
            Each exemplar represents a macroclass.

            >>> classes
            {"a":["a","A","aa"], "b":["b","B","BBB"]}

    """
    grouped = paradigms.fillna(0).groupby(list(paradigms.columns))
    mc = {}

    for name, group in grouped:
        members = list(group.index)
        exemplar = min(members, key=lambda string: len(string))
        mc[exemplar] = members

    return mc


def find_min_attribute(tree, attr):
    """Find the minimum value for an attribute in a tree.

    Arguments:
        tree (Node): The tree in which to find the minimum attribute.
        attr (str): the attribute's key."""
    agenda = [tree]
    mini = np.inf
    while agenda:
        node = agenda.pop(0)
        if node.children:
            agenda.extend(node.children)
        if attr in node.attributes and float(node.attributes[attr]) < mini:
            mini = node.attributes[attr]

    return mini
