from flask import Flask, render_template, request
import pandas as pd
#from bokeh.charts import Histogram
from bokeh.embed import components
from bokeh.plotting import figure
import numpy as np
from collections import defaultdict
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import magma


app = Flask(__name__)

feature_names = ["3","15","30"]

# Create the main plot
def create_figure(n):
        
        mass_spec = defaultdict(list)
        int_fict=np.arange(50)
        order_loc=np.ones(50)

        n=int(n)
        choices=['Todos','Ninguno','Retroexcavdora, Camion']
        lista_zonas=[(1.0,'Zona '+str(k),np.random.choice(choices)) for k in range(n)]     
        
        for scale, mz, evt_list in lista_zonas:
            mass_spec["RT"].append(order_loc)
            order_loc=order_loc+1
            mass_spec["RT_intensity"].append(int_fict*scale)#norm(loc=120.4).pdf(RT_x) * 
            mass_spec['Eventos_permitidos'].append(evt_list)
            mass_spec['Intensity_tip'].append(mz)
            mass_spec['rel_int'].append(np.random.normal(loc=0.9,scale=0.01))
        mass_spec['color'] = magma(n)

        source = ColumnDataSource(mass_spec)

        x_limit_left=n+0.1
        
        p = figure(plot_width=800, plot_height=600,x_range=(0,x_limit_left))
        hover_t=HoverTool(show_arrow=False, tooltips=[
            ('Eventos permitidos', '@Eventos_permitidos'),
            ('Intensidad relativa', '@rel_int')
        ])
        p.add_tools(hover_t)
        p.image_url(url=['static/mapa.png'], x=0, y=0, w=x_limit_left, h=49,anchor='bottom_left')
        ml=p.multi_line(xs='RT', ys='RT_intensity', legend="Intensity_tip",
                     line_width=5, line_color='color', line_alpha=0.6,
                     hover_line_color='pink', hover_line_alpha=1.0,
                     source=source)
        p.xaxis.bounds=(0,n)
        p.yaxis.visible=False

        
        return p

# Index page
@app.route('/')
def index():
        # Determine the selected feature
        current_feature_name = request.args.get("feature_name")
        if current_feature_name == None:
                current_feature_name = "15"

        # Create the plot
        plot = create_figure(current_feature_name)
                
        # Embed plot into HTML via Flask Render
        script, div = components(plot)
        return render_template("iris_index3.html", script=script, div=div,
                feature_names=feature_names,  current_feature_name=current_feature_name)

# With debug=True, Flask server will auto-reload 
# when there are code changes
if __name__ == '__main__':
        app.run(port=5000, debug=True)
