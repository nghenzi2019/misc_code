from collections import defaultdict

import numpy as np
from scipy.stats import norm

from bokeh.plotting import show, figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import magma

RT_x = np.linspace(118, 123, num=50)

mass_spec = defaultdict(list)
i=np.arange(50)
ll=np.ones(50)
for scale, mz, evt_list in [(1.0, 'ZONA 1', 'Ninguno'), (1.0, 'ZONA 2', 'Todos'), (1.0, 'ZONA 3', 'Todos'), (1.0, 'ZONA 4', 'Retro,excavacion,camion'), (1.0, 'ZONA 5', 'Todos'), (1.0, 'ZONA 6', 'Todos')]:
    mass_spec["RT"].append(ll)
    ll=ll+1
    mass_spec["RT_intensity"].append(i*scale)#norm(loc=120.4).pdf(RT_x) * 
    mass_spec['Eventos_permitidos'].append(evt_list)
    mass_spec['Intensity_tip'].append(mz)
    mass_spec['rel_int'].append(np.random.normal(loc=0.9,scale=0.01))
mass_spec['color'] = magma(6)

source = ColumnDataSource(mass_spec)

p = figure(plot_width=800, plot_height=600,x_range=(0,6.1))
hover_t=HoverTool(show_arrow=False, tooltips=[
    ('Eventos permitidos', '@Eventos_permitidos'),
    ('Intensidad relativa', '@rel_int')
])
p.add_tools(hover_t)
p.image_url(url=['mapa.png'], x=0, y=0, w=6.1, h=49,anchor='bottom_left')
ml=p.multi_line(xs='RT', ys='RT_intensity', legend="Intensity_tip",
             line_width=5, line_color='color', line_alpha=0.6,
             hover_line_color='pink', hover_line_alpha=1.0,
             source=source)

#, line_policy='next'
N = 500
x = np.linspace(0, 10, N)
y = np.linspace(0, 10, N)
xx, yy = np.meshgrid(x, y)
d = np.sin(xx)*np.cos(yy)

#p.image(image=[d], x=0, y=0, dw=10, dh=10, palette="Spectral11")
p.xaxis.bounds=(0,6)
p.yaxis.visible=False
show(p)
