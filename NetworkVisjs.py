from __future__ import division

import numpy as np

from bokeh.core.properties import Instance, String, Int
from bokeh.models import ColumnDataSource, LayoutDOM
from bokeh.io import show

JS_CODE = """
import * as p from "core/properties"
import { LayoutDOM, LayoutDOMView } from "models/layouts/layout_dom"
OPTIONS =
    interaction: {
        hover: true,
        selectConnectedEdges: true,
        hoverConnectedEdges: true
    }
    manipulation: {
        enabled: false
    }
    nodes: {
        shape: 'dot',
        size: 20,
        font: {
            strokeWidth: 1.5
        }
    }
    edges: {
        color: {
            inherit: "both"
        }
    }

export class NetworkView extends LayoutDOMView

    initialize: (options) ->
        super(options)

        url = "https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"

        script = document.createElement('script')
        script.src = url
        script.async = false
        script.onreadystatechange = script.onload = () => @_init()
        document.querySelector("head").appendChild(script)

    _init: () ->
        nodes = new vis.DataSet();
        edges = new vis.DataSet();
        data = {
            nodes: nodes,
            edges: edges
        };
        @get_data(nodes,edges);
        @_graph = new vis.Network(@el, data, OPTIONS)
        @connect(@model.data_source.change, () => @get_data(nodes,edges))
        @_graph.on("doubleClick", (params) => @fun(params));

    get_data: (nodes, edges) ->
        nodes.clear()
        edges.clear()
        source = @model.data_source
        for i in [0...source.get_length()]
            nodes.add({
                id: i,
                label: source.get_column(@model.label)[i],
                color: source.get_column(@model.color)[i],
                title: source.get_column(@model.label)[i]
            })
            for j in [0...source.get_column(@model.edges)[i].length]
                edges.add({
                    from: i,
                    to: source.get_column(@model.edges)[i][j]-1,
                    title: String((source.get_column(@model.values)[i][j]*10).toFixed 2),
                    value: (source.get_column(@model.values)[i][j]*10).toFixed 2
                })
    
    fun: (params) ->
        if params.nodes
            @model.selected = params.nodes[0]
    
        

export class Network extends LayoutDOM

    default_view: NetworkView

    type: "Network"

    @define {
        label: [p.String]
        edges: [p.String]
        values: [p.String]
        color: [p.String]
        data_source: [p.Instance]
        selected: [p.Number]
    }
"""

# This custom extension model will have a DOM view that should layout-able in
# Bokeh layouts, so use ``LayoutDOM`` as the base class. If you wanted to create
# a custom tool, you could inherit from ``Tool``, or from ``Glyph`` if you
# wanted to create a custom glyph, etc.


class Network(LayoutDOM):
    __implementation__ = JS_CODE
    data_source = Instance(ColumnDataSource)
    label = String
    edges = String
    values = String
    color = String
    selected = Int(default=-1)
