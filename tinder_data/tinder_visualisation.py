"""
Created on Wed Jul 24 19:07:18 2019
@author: rian-van-den-ander
"""


#! pip install pywaffle
# Reference: https://stackoverflow.com/questions/41400136/how-to-do-waffle-charts-in-python-square-piechart
from pywaffle import Waffle
import matplotlib.pyplot as plt

fig = plt.figure(
    FigureClass=Waffle,
    plots={
        '111': {
                
            'values': [31740,(7274-349),(349-223),(223-80),(80-25),(25-3),3],
            'labels': ["Swipes","right swipes","matches","conversations initiated","meaningful conversations","dates","connection"],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 12},
            'title': {'label': 'Test', 'loc': 'center', 'fontsize':18}
        },
    },
    rows=20,
    colors= ['grey','firebrick','mediumseagreen','g','dodgerblue','slateblue','deeppink'],
    figsize=(375, 20),
    icons='child', icon_size=11,
    icon_legend=True
)
