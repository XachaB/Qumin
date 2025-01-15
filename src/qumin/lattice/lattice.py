# !usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict
from ..clustering.node import Node
from os.path import join, dirname
import logging

matplotlib.use("agg", force=True)
log = logging.getLogger()

try:
    import mpld3
except:
    mpld3 = None
    log.warning("Warning: mpld3 could not be imported. No html export possible.")
from concepts import Context
import pandas as pd
from tqdm import tqdm

axes = {'facecolor': 'None', 'edgecolor': 'None', 'linewidth': 0}
grid = {'alpha': 0, 'linewidth': 0}
lines = {'linewidth': 0.5}

matplotlib.rc('lines', **lines)
matplotlib.rc('axes', **axes)
matplotlib.rc('grid', **grid)


def _load_external_text(filename):
    return "\n".join(
        open(join(dirname(__file__), filename), "r", encoding="utf-8").readlines())


def _node_to_label_IC(node, comp=None, **kwargs):
    objs = node.attributes.get("objects", node.labels)
    if not objs:
        objs = node.labels
    size = str(node.attributes.get("size", "unknown nb of")) + " lexèmes"
    header = "<table><thead><th colspan=2> Ex:" + objs[0] + ", " + size + " </th></thead>"
    if "common" in node.attributes:
        line = "<tr><th>{}</th><td>{}</td></tr>"
        line2 = "<tr class='alternate'><th>{}</th><td>{}</td></tr>"
        line_no_head = "<tr><td colspan=2>{}</td></tr>"
        common = ""
        for properties in node.attributes["common"]:
            for prop in properties.split(";"):
                if "=" in prop:
                    attr, val = prop.split("=")
                    val = val.split("/")[0]
                    if comp and attr.startswith(comp):
                        common += line2.format(attr[len(comp):], val)
                    else:
                        common += line.format(attr, val)
                else:
                    common += line_no_head.format(prop)
        if not common:
            common = "<tr><td colspan=2>Empty</td></tr>"

        return header + common + "</table>"
    return ""


class ICLattice(object):
    """Inflection Class Lattice.

    This is a wrapper around (:class:`concepts.Context`).
    """

    def __init__(self, patterns, leaves, annotate=None, comp_prefix=None, aoc=False,
                 **kwargs):
        """
        Arguments:
            patterns (patterns.ParadigmPatterns): patterns
            leaves (dict): Dictionnaire de microclasses
            annotate (dict): Extra annotations to add on lattice.
                Of the form: {<object label>:<annotation>}
            aoc (bool): Whether to limit ourselves to Attribute or Object Concepts.
            kwargs: all other keyword arguments are passed to table_to_context
        """

        self.comp = comp_prefix  # whether there are two sets of properties.
        incidence_table = patterns.incidence_table()
        self.context = Context.fromstring(incidence_table.to_csv(), frmat='csv')
        self.lattice = self.context.lattice
        if annotate:
            for label in annotate:
                if label in self.lattice.supremum.extent:
                    self.lattice[[label]]._extra_qumin_annotation = annotate[label]

        self.leaves = leaves
        log.debug("Converting to qumin node...")
        if aoc:
            self.nodes = self._lattice_to_nodeAOC()
        else:
            self.nodes = self._lattice_to_node()
        font = {'family': 'DejaVu Sans',
                'weight': 'normal',
                'size': 9}
        matplotlib.rc('font', **font)

    def _pat_range(self):
        mini = maxi = 1
        for extent, intent in self.lattice:
            l = len(self.lattice[intent].properties)
            if l < mini:
                mini = l
            elif l > maxi:
                maxi = l
        return mini, maxi

    def _lattice_to_node(self, keep_infimum=False):
        def make_nodes(concepts, prb):
            nodes = {}
            for concept in concepts:
                extent = concept.extent
                intent = concept.intent
                properties = concept.properties
                objects = concept.objects
                size = sum(
                    len(self.leaves[label]) for label in extent if label in self.leaves)
                annotations = getattr(concept, '_extra_qumin_annotation', {})
                nodes[extent] = Node(extent, intent=intent, size=size, common=properties,
                                     objects=objects,
                                     macroclass=False, **annotations)
                prb.update(1)
            return nodes

        concepts = sorted([v for v in self.lattice if keep_infimum or v.extent != ()],
                          key=lambda x: len(x.extent), reverse=True)

        with tqdm(total=len(concepts) * 2) as prb:
            # Creating nodes
            nodes = make_nodes(concepts, prb)

            # Creating arcs
            for vertice in concepts:
                for daughter in vertice.lower_neighbors:
                    if keep_infimum or daughter.extent != ():
                        nodes[vertice.extent].children.append(nodes[daughter.extent])
                prb.update(1)
        root = nodes[self.lattice.supremum.extent]
        return root

    def _lattice_to_nodeAOC(self):
        def make_nodes(concepts, prb):
            nodes = {}
            for concept in concepts:
                extent = concept.extent
                intent = concept.intent
                properties = concept.properties
                objects = concept.objects
                size = sum(
                    len(self.leaves[label]) for label in extent if label in self.leaves)
                annotations = getattr(concept, '_extra_qumin_annotation', {})
                nodes[extent] = Node(extent, intent=intent, size=size, common=properties,
                                     objects=objects,
                                     macroclass=False, **annotations)
                prb.update(1)
            return nodes

        # nodes = self._get_nodes(keep_infimum=False)
        supremum = self.lattice.supremum
        infimum = self.lattice.infimum

        def concept_sorter(concept):
            return (len(concept.extent), -len(list(concept.upset())))

        # select concept in AOC
        aoc = {c for c in self.lattice
               if (c == supremum or c.properties or c.objects) and c != infimum}

        # Make links (long way)
        concepts = sorted(aoc, key=concept_sorter, reverse=True)
        downsets = {c: set(c.downset()) for c in self.lattice}
        children = defaultdict(set)
        l = len(concepts)
        with tqdm(total=l * 3) as prb:
            # Compute descendants in aoc poset
            for i in range(l):
                concept = concepts[i]
                span = set()
                for candidate in sorted((downsets[concept] & aoc) - {concept},
                                        key=concept_sorter, reverse=True):
                    if candidate not in span:
                        children[concept.extent].add(candidate)
                        span.update(downsets[candidate])
                prb.update(1)

            # Make and link Node objects
            # Creating nodes
            nodes = make_nodes(concepts, prb)
            # Creating arcs
            for vertice in concepts:
                for daughter in children[vertice.extent]:
                    nodes[vertice.extent].children.append(nodes[daughter.extent])
                prb.update(1)
            return nodes[supremum.extent]

    def parents(self, identifier):
        """Return all direct parents of a node  which corresponds to the identifier."""
        return list(self.lattice[identifier].upper_neighbors)

    def ancestors(self, identifier):
        """Return all ancestors of a node  which corresponds to the identifier."""
        concept = self.lattice[identifier]
        return [c for c in concept.upset() if c != concept]

    def stats(self):
        """Returns some stats about the classification size and shape.
        Based on self.nodes, not self.lattice: stats are different depending on AOC/not AOC.
        """

        def height(node):
            if not node.children:
                node.attributes["height"] = 1
                return 1
            else:
                if "height" in node.attributes:
                    return node.attributes["height"]
                h = max(height(child) for child in node.children) + 1
                node.attributes["height"] = h
                return h

        nb_arcs = sum(len(x.children) for x in self.nodes)
        nb_noeuds = len([x for x in self.nodes])
        stats_lattice = {"Microclasses": len(self.leaves),
                         "Base": len(self.lattice.atoms),
                         "Hauteur": height(self.nodes),
                         "Degré": nb_arcs / (nb_noeuds - 2),
                         # -2 car on ignore supremum et infimum
                         "Noeuds": nb_noeuds - 1  # -1 car on ignore infimum
                         }

        if self.comp:
            left = 0
            right = 0
            both = 0
            for node in self.nodes:
                cmp = sum(att.startswith(self.comp) for att in node.attributes["common"])
                if cmp > 0:
                    if cmp < len(node.attributes["common"]):
                        both += 1
                    else:
                        left += 1
                else:
                    right += 1
            log.info("Concepts définissant des propriétés "
                     "de la classification de gauche (-b): %s", left)
            log.info("Concepts définissant des propriétés "
                     "de la classification de droite: %s", right)
            log.info("Concepts définissant des propriétés "
                     "des deux classifications: %s", both)
        return pd.Series(stats_lattice,
                         index=["Microclasses", "Base", "Hauteur", "Degré", "Noeuds"])

    def _draw_one(self, node, figsize=(24, 12), scale=False, colormap="Blues", point=None,
                  **kwargs):
        mini, maxi = self._pat_range()
        cm = matplotlib.cm.get_cmap(colormap)
        cnorm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        smap = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cm)
        colors = ['#444444', '#aaaaaa']

        def custom_zorder(node):
            return len(node.attributes["common"])

        def leaves_label(node):
            n = " ({})".format(str(node.attributes.get("size", 1)))
            return ", ".join(node.labels) + n

        def point_function(node):
            default = {"color": colors[0],
                       "edgecolors": colors[0],
                       "zorder": 3,
                       "marker": matplotlib.markers.MarkerStyle(marker="o")}

            if self.comp:
                cmp = sum(att.startswith(self.comp) for att in node.attributes["common"])
                if cmp > 0:  # bicolor marker
                    if cmp < len(node.attributes["common"]):
                        default["facecolor"] = colors[1]
                        default["linewidth"] = 3
                        del default["color"]
                    else:  # marker in color 2
                        default["facecolor"] = colors[1]
                        default["edgecolor"] = colors[1]
                        del default["color"]
            default["s"] = 20 + ((node.attributes.get("size", 0) + 1) / (
                    self.nodes.attributes.get("size", 0) + 1)) * 100
            node.attributes["point_settings"] = default
            return default

        def default_edge_attr(node, child):
            return {"color": colors[0], "zorder": custom_zorder(node)}

        params = dict(leavesfunc=leaves_label,
                      nodefunc=lambda n: "",
                      edge_attributes=default_edge_attr,
                      point=point_function if point else None,
                      horizontal=False, square=False, layout="qumin",
                      )
        params.update(kwargs)
        fig = plt.figure(figsize=figsize)  # for export: 12,6
        lines, ordered_nodes = node.draw(**params)

        if scale:
            colors = [smap.to_rgba(i) for i in range(mini, maxi)]
            smap.set_array(colors)
            plt.colorbar(smap, norm=cnorm)
        return fig, lines, ordered_nodes

    def draw(self, filename, title="Lattice", **kwargs):
        """Draw the lattice using :class:`qumin.clustering.node.Node`'s drawing function."""
        fig, lines, ordered_nodes = self._draw_one(self.nodes, **kwargs)
        if title is not None:
            fig.suptitle(title)
        log.info("Drawing figure to: {}".format(filename))
        axis = plt.gca()
        axis.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        axis.xaxis.set_major_locator(plt.NullLocator())
        axis.yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def to_html(self, filename, node_formatter=_node_to_label_IC, **kwargs):
        """Draw an interactive lattice using :class:`qumin.clustering.node.Node`'s drawing function and mpld3.

        Arguments:
            filename (str): filename of the exported html page.
            node_formatter (Callable): custom function to format nodes
        """
        css = _load_external_text("table.css")
        fig, lines, ordered_nodes = self._draw_one(self.nodes,
                                                   figsize=(20, 9),
                                                   n=4,
                                                   scale=False,
                                                   point={"s": 50},
                                                   # TODO: Something wrong here
                                                   interactive=True, **kwargs)

        paths = list(
            filter(lambda obj: type(obj) is matplotlib.collections.PathCollection,
                   fig.axes[0].get_children(), ))
        lines = list(filter(lambda obj: type(obj) is matplotlib.lines.Line2D and
                                        len(obj.get_xdata(orig=True)) > 1,
                            fig.axes[0].get_children()))
        points_ids = []
        corrd_to_points = {}

        for p, v in zip(paths, ordered_nodes):
            x, y = v.attributes["_x"], v.attributes["_y"]
            p_id = mpld3.utils.get_id(p)  # ,"pts")
            corrd_to_points[(x, y)] = p_id
            label = node_formatter(v, comp=self.comp)
            points_ids.append(p_id)
            tooltip = mpld3.plugins.PointHTMLTooltip(p, [label], css=css)
            mpld3.plugins.connect(fig, tooltip)

        point_to_artists = defaultdict(set)
        lines = sorted(lines, key=lambda l: min(l.get_ydata()))
        for l in lines:
            parent, child = zip(*l.get_data(orig=True))
            childp = corrd_to_points[tuple(child)]
            parentp = corrd_to_points[tuple(parent)]

            if 0 in child:
                point_to_artists[childp] = set()
            point_to_artists[parentp].add(childp)
            point_to_artists[parentp].add(mpld3.utils.get_id(l))
            point_to_artists[parentp].update(point_to_artists[childp])

        point_to_artists = {p: list(point_to_artists[p]) for p in point_to_artists}

        # root = corrd_to_points[(node.attributes["_x"],node.attributes["_y"])]
        mpld3.plugins.connect(fig, _HighlightSubTrees(points_ids, dict(point_to_artists)))
        mpld3.save_html(fig, filename, template_type="simple")


if not mpld3:
    def to_html_disabled(*args, **kwargs):
        log.warning("mpld3 could not be imported. No html export possible.")

    ICLattice.to_html = to_html_disabled
else:
    class _HighlightSubTrees(mpld3.plugins.PluginBase):
        """A plugin to highlight lines on hover"""

        JAVASCRIPT = _load_external_text("HighlightSubTrees.js")

        def __init__(self, points_ids, point_to_artists):
            self.css_ = """
                        path.unfocus
                        {-webkit-transition: all 0.5s ease;
                        -moz-transition: all 0.5s ease;
                        -o-transition: all 0.5s ease;
                        transition: all 0.5s ease;
                        fill-opacity: 0.1 !important;
                        stroke-opacity:0.1 !important;}

                        .mpld3-ygrid, .mpld3-xgrid, .mpld3-yaxis, .mpld3-xaxis
                        {display: none !important}
                        """

            self.dict_ = {"type": "highlightsubtrees",
                          "points_ids": points_ids,
                          "points_to_artist": point_to_artists,
                          "min": 0.2,
                          "max": 1}
