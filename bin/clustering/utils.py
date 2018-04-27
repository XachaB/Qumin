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
            color (str): color of the splines from this node to its children,
              in a format usable by pyplot.
              Currently, red ("r") is used when the node didn't decrease
              Description length, blue ("b") otherwise.
            macroclass (bool): Is the node in a macroclass ?
            macroclass_root (bool): Is the node the root of a macroclass ?

            The attributes "_x" and "_rank" are reserved,
            and will be overwritten by the draw function.
    """

    def __init__(self, labels, children=None, **kwargs):
        """Node constructor.

        Arguments:
            labels (list): labels of all the leaves under this node.
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

    def __str__(self):
        """Return a repr string for Nodes."""
        attrs = " - ".join("{}={}".format(key, self.attributes[key]) for key in self.attributes)
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
                child_macroclasses = child.macroclasses(parent_is_macroclass=self_is_macroclass)
                macroclasses_under.update(child_macroclasses)
            return macroclasses_under
        return {}

    def _recursive_xy(self):
        if self.attributes.get("_y",None) is None:
            y = 1
            if len(self.children) > 0:
                xs,ys = zip(*[child._recursive_xy() for child in self.children])
                y += max(ys)
                x = sum(xs)/len(self.children)
                self.attributes["_x"] = x
            self.attributes["_y"] = y
        return self.attributes["_x"],self.attributes["_y"]

    def _erase_xy(self):
        self.attributes["_x"] = None
        self.attributes["_y"] = None
        for child in self.children:
            child._erase_xy()

    def compute_xy(self, tree_placement=False, pos=False):
        if "_y" in self.attributes and self.attributes["_y"] is not None :
            self._erase_xy()

        if tree_placement: # For trees
            leaves = self.leaves()
            x = 0
            step = 10
            for leaf in leaves:
                leaf.attributes["_x"] = x
                x += step
            self._recursive_xy()
        else:
            if not pos:
                pos = nx.drawing.nx_agraph.graphviz_layout(self.to_networkx(), prog='dot')
            for node in self:
                node.attributes["_x"], node.attributes["_y"] = pos[tuple(node.labels)]

    def to_networkx(self):
        if not nx_loaded:
            raise ImportError("Can't convert to networkx, because it couldn't be loaded.")
        G=nx.DiGraph()
        for node in self:
            name = tuple(node.labels)
            G.add_node(name,**node.attributes)
            for child in node.children:
                childname = tuple(child.labels)
                G.add_edge(name,childname)
        return G

    def __iter__(self):
        visited = set()
        agenda = [self]
        while agenda:
            node = agenda.pop(0)
            if node not in visited:
                visited.add(node)
                agenda = node.children + agenda
                yield node

    def leaves(self):
        return filter(lambda x:not bool(x.children),self)

    def __repr__(self):
        """Return the inflection class tree as a string with parenthesis.

        Assumes size, DL and color attributes,
        with color = "r" if this is above a macroclass.

        Example:
            In the label, fields are separated by "#" as such::

                (<labels>#<size>#<DL>#<color> )
        """
        labels = "&".join(self.labels)
        ignore = ["_rank", "_x", "macroclass", "macroclass_root"]
        attributes = "#".join("{}={}".format(key, str(self.attributes[key]).replace(" ", "_")) for key in sorted(self.attributes) if key not in ignore)
        children_str = [repr(child) for child in self.children]
        string = "(" + labels + "#" + attributes + " ".join([""] + children_str) + " )"

        return string

    def _recursive_to_latex(self, n, nodelabel=None, macroclasses=False, leavesfunc=None):
        """Return the nodes of a latex string.

        Arguments:
            n : depth of the node (affects indents).
            nodelabel: The name of the attribute to write on the nodes.
                Defaults to "size". No attribute written if its value is None.
            macroclasses: If True, the macroclasses will be bolded,
                and the arcs above them will be gray.
            leavesfunc (fun):
                A function that will be applied to leaves
                before writing them down. Takes a Node, returns a str.
        """
        if not nodelabel:
            def nodelabel(node):
                return node.attributes["size"]
        if not leavesfunc:
            def leavesfunc(n):
                return n.labels[0]
        if n > 0 and macroclasses :
            if self.attributes["macroclass"] :
                style = "\edge[color=black!20]; "
            else:
                style = "\edge[thick]; "
        else:
            # root
            style = ""

        fill_values = {"tabs": "    " * (n+2), "style": style}
        fill_values["comment"] = ", ".join("{}={}".format(key, self.attributes[key]) for key in self.attributes)
        if self.children and len(self.children) > 0:
            template = "{tabs}{style}[.{{{label}}} %{comment}\n{children}{tabs}]\n"

            fill_values["label"] = nodelabel(self)

            children = (child._recursive_to_latex(n+1, nodelabel=nodelabel,
                                                  leavesfunc=leavesfunc)
                        for child in self.children)

            fill_values["children"] = "".join(sorted(children,key=len))
        else:
            template = "{tabs}{style}[.{{{label}}} ]%{comment}\n"
            fill_values["label"] = leavesfunc(self)

        return template.format(**fill_values)

    def to_latex(self, nodelabel=None, vertical=True, level_dist=50,
                 square=True, leavesfunc=lambda n: n.labels[0], scale=1):
        """Return a latex string, compatible with tikz-qtree

        Arguments:
            nodelabel: The name of the attribute to write on the nodes.
            vertical: Should the tree be drawn vertically ?
            level_dist: Distance between levels.
            square: Should the arcs have a squared shape ?
            leavesfunc (fun):
                A function that will be applied to leaves
                before writing them down. Takes a Node, returns a str.
        """
        tikzset = ["level distance={}pt, ".format(level_dist)]
        if vertical and square:
            tikzset += ["edge from parent/.style={draw,",
                        "\n\t\tedge from parent path={(\\tikzparentnode.south)",
                        "\n\t\t-- +(0,-8pt)",
                        "\n\t\t-| (\\tikzchildnode)}}"]
        if not vertical:
            tikzset += ["execute at begin node=\\strut, ",
                        "grow'=right, ",
                        "every tree node/.style={anchor=base west}, "]

            if square:
                tikzset += ["edge from parent/.style={draw,",
                            "\n\t\tedge from parent path={(\\tikzparentnode.east)",
                            "\n\t\t-- +(0.2,0) |- (\\tikzchildnode.west)}}"]

        strings = ["\\begin{tikzpicture}[scale=",
                   str(scale),
                   "]\n",
                   "\t\\tikzset{",
                   "".join(tikzset),
                   "}",
                   "\n\t\t\\Tree",
                   # trim first indent, it's before "\Tree" already
                   self._recursive_to_latex(0, nodelabel=nodelabel,
                                            leavesfunc=leavesfunc)[2:],
                   "\\end{tikzpicture}"]

        return "".join(strings)

    def draw(self, horizontal=False, square=False,
             leavesfunc=lambda n: n.labels[0],
            nodefunc=None,
             keep_above_macroclass=True,
             annotateOnlyMacroclasses=False, point=None,
             edge_attributes=None, interactive=False, lattice=False, pos=None):
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
                Should the edges above macroclasses be drawn ?
                (Defaults to True).
            annotateOnlyMacroclasses : If `True` and nodelabel isn't `None`,
                only the macroclasses nodes are annotated.
            point (fun):
                A function that maps a node to point attributes.
            edge_attributes (fun):
                A function that maps a pair of nodes to edge attributes.
                    By default, use the parent's color and "-" linestyle for nodes,
                    "--" for leaves.
            interactive (bool):
                Whether this is destined to create an interactive plot.
            lattice (bool):
                Whether this node is a lattice rather than a tree.
        """
        tree_placement= not lattice

        self.compute_xy(pos=pos,tree_placement=tree_placement)

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
                attributes = {"linestyle": "-" if node.children else "--"}
                attributes["color"] = node.attributes.get("color","#333333")
                if edge_attributes is not None:
                    attributes.update(edge_attributes(node, child))
                return attributes

            if horizontal:
                leafoffset = (-5,0)
                textoffset = (5,0)
                lva = "center"
                lha = "right"
                va = "center"
                ha = "left"
                r= 0
                def coords(node):
                    return node.attributes["_y"], node.attributes["_x"]
            else:
                lva = "top"
                lha = "right"
                va = "bottom"
                ha = "center"
                leafoffset = (0,-3)
                textoffset = (0,5)
                r= 45
                def coords(node):
                    return node.attributes["_x"], node.attributes["_y"]

            ax = plt.gca()
            bg = ax.patch.get_facecolor()

            lines = []
            all_nodes = []

            for node in self:
                this_x,this_y = coords(node)

                # Write the text
                if tree_placement:
                    microclass = node.labels if not node.children else None
                else:
                    microclass = node.attributes.get("objects",None)

                # Plot the arcs
                for child in node.children:
                    child_x, child_y = coords(child)
                    attr = default_edge_attributes(node,child)
                    if square:
                        if horizontal:
                            l = ax.plot((this_x,this_x,child_x),(this_y,child_y,child_y), **attr)
                        else:
                            l = ax.plot((this_x,child_x,child_x),(this_y,this_y,child_y), **attr)
                    else:
                        l = ax.plot((this_x,child_x),(this_y,child_y), **attr)
                    lines.extend(l)

                # Plot the point
                if point is not None:
                    coll = ax.scatter((this_x,), (this_y,), **point(node))
                    lines.append(coll)

                # Write annotations
                if microclass:
                    tmp = node.labels
                    node.labels = list(microclass)
                    plt.annotate(leavesfunc(node), xy=(this_x, this_y), xycoords='data', va=lva,ha=lha,rotation=r)
                    node.labels = tmp
                else:
                    text = annotate(node)
                    if text is not None:
                        plt.annotate(text, xy=(this_x,this_y),va=va, ha=ha,textcoords='offset points', xytext=textoffset)

                all_nodes.append(node)

            # Scale axes
            ax.autoscale()

            if interactive:
                if horizontal:
                    ax.margins(x=0.3,y=0.1)
                else:
                    ax.margins(y=0.3,x=0.1)

            plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks
                top='off',         # ticks along the top edge
                bottom='off',       # ticks along the bottom edge
                right='off',      # ticks along the right edge
                left='off',      # ticks along the left edge
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
                    pass # only convert numbers


            stack.append(Node(labels, **attributes))

        if item[0] == ")":
            if len(item) > 1:
                print("Warning, bad format ! #{}#".format(item))
            if len(stack) > 1:
                child = stack.pop(-1)
                parent = stack[-1]
                # Macroclass are one level below the red :
                child.attributes['macroclass_root'] = parent.attributes['color'] == 'r' and \
                                                      child.attributes['color'] != 'r'
                stack[-1].children.append(child)
            else:
                return stack[0]

    if len(stack) > 1:
        print("Warning, unmatched parenthesis or no root ! ", stack)

    print(stack[0])
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
    grouped = paradigms.groupby(list(paradigms.columns))
    mc = {}

    for name, group in grouped:
        members = list(group.index)
        exemplar = min(members, key=lambda string: len(string))
        mc[exemplar] = members

    return mc


def label_function(labels, classes_size):
    """Return labels in the form "lemma (size)"."""
    template = "{} ({})"
    names = ", ".join(labels)
    size = str(sum(classes_size[label] for label in labels))
    return template.format(names, size)


def find_min_attribute(tree, attr):
    """Find the minimum value for an attribute in a tree.

    Arguments:
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
