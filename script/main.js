var margin = {top: 50, right: 50, bottom: 50, left: 50};
var width = 600 - margin.left - margin.right;
var height = 600 - margin.top - margin.bottom;

var xScale = d3.scaleLinear().domain([0, 1]).range([margin.left, width - margin.right]);
var yScale = d3.scaleLinear().domain([0, 1]).range([height - margin.bottom, margin.top]);
          
var svg = d3.select("body")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            
var line_plot = d3.line()
              .x(function(d) {return xScale(d.fpr);})
              .y(function(d) {return yScale(d.tpr);});
      
var xAxis = svg.append("g")
               .attr("transform", `translate(0, ${height - margin.bottom})`)
               .call(d3.axisBottom().scale(xScale));

var yAxis = svg.append("g")
               .attr("transform", `translate(${margin.left}, 0)`)
               .call(d3.axisLeft().scale(yScale));
               
svg.append("text")
   .attr("y", 500)
   .attr("x", 235)
   .style("text-anchor", "middle")
   .text("False Positive Rate");
   
svg.append("text")
   .attr("transform", "rotate(-90)")
   .attr("y", 20)
   .attr("x", -240)
   .style("text-anchor", "middle")
   .text("True Positive Rate");
               
svg.append("g")
   .append("path")
   .attr('stroke', 'black')
   .attr('stroke-width', '1.5px')
   .attr('fill', 'none')
   .attr("class", "Reference")
   .style("stroke-dasharray", ("3, 3"))
   .attr("d", line_plot([{"fpr":+0, "tpr":+0}, {"fpr":+1, "tpr":+1}]));
   
// Visualization
function Visualization(data){
	svg.append("path")
	   .attr('stroke', 'red')
	   .attr('stroke-width', '1.5px')
	   .attr('fill', 'none')
	   .attr("class", "ROC")
	   .attr("d", line_plot(data));
}

// Read data from server and draw line chart
function Submit() {
	var url = "http://localhost:5000/";
	var radioList = document.getElementsByName("preprocessing");
		for(var i=0; i<radioList.length; i++){
			if(radioList[i].checked){
				var preprocessing = radioList[i].value;
			}
		}
	var C = document.getElementById("c").value;
	C = parseFloat(C).toFixed(2);
	url = url.concat(preprocessing).concat("/").concat(C);
	d3.json(url, function(data){
		console.log(d3.selectAll(".ROC").remove());
		Visualization(data);
	})
}