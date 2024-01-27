mpld3.register_plugin("highlightsubtrees", HighlightSubTrees);
HighlightSubTrees.prototype = Object.create(mpld3.Plugin.prototype);
HighlightSubTrees.prototype.constructor = HighlightSubTrees;
HighlightSubTrees.prototype.requiredProps = ["points_ids","points_to_artist","min","max"];

function HighlightSubTrees(fig, props){
    mpld3.Plugin.call(this, fig, props);
};



HighlightSubTrees.prototype.draw = function(){
  var min = this.props.min;
  var max = this.props.max;
  for(var i=0; i<this.props.points_ids.length; i++){
     var id = this.props.points_ids[i];
     var obj = mpld3.get_element(id, this.fig);

     obj.pathsobj.property("__subtree__",
         this.props.points_to_artist[id]);

    function toggle_alpha(selection,alpha){
        selection.style("stroke-opacity",alpha);
        selection.style("fill-opacity",alpha);
    };

     obj.elements()
         .on("mousedown", function(d, i){

                var element = d3.select(this);
                if(element.style("stroke-opacity") < 1){
                    alpha = 1;
                }
                else {
                    alpha = 0.1;
                }
                toggle_alpha(element,alpha);
                var subtree = element.property("__subtree__");

                for(var j=0; j<subtree.length; j++){
                    var e = mpld3.get_element(subtree[j],this.fig);
                    var selection = [];
                    if(e['path'] !== undefined){
                        selection = e.path;
                    }
                    else if(e['pathsobj'] !== undefined){
                        selection = e.pathsobj;
                    }
                    toggle_alpha(selection,alpha);

                };
                });
}};