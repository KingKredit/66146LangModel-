import solara as sl

from modelv2 import LanguageModel


import solara as sl
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from modelv2 import LanguageModel
import matplotlib.pyplot as plt
import networkx as nx


LANG_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def language_to_color(lang_id: int) -> str:
    # map integer ID into [0, 1] for a smooth colormap
    x = (lang_id % 256) / 256  # still wraps eventually, but has many more unique colors
    rgba = cm.get_cmap("hsv")(x)
    return mcolors.to_hex(rgba)

def agent_portrayal(agent):
    color = language_to_color(int(agent.language))
    return AgentPortrayalStyle(
        color=color,
        marker="o",
        size=40,
    )

# def agent_portrayal(agent):
#     # map language index -> color
#     idx = int(agent.language) % len(LANG_COLORS)
#     return AgentPortrayalStyle(
#         color=LANG_COLORS[idx],
#         marker="o",
#         size=40,
#     )


model_params = {
    "numAgents": {
        "type": "SliderInt",
        "value": 100,
        "label": "Number of agents",
        "min": 10,
        "max": 500,
        "step": 10,
    },
    "width": {
        "type": "SliderInt",
        "value": 20,
        "label": "Grid width",
        "min": 5,
        "max": 80,
        "step": 1,
    },
    "height": {
        "type": "SliderInt",
        "value": 20,
        "label": "Grid height",
        "min": 5,
        "max": 80,
        "step": 1,
    },
    "r": {
        "type": "SliderInt",
        "value": 1,
        "label": "Interaction radius r",
        "min": 1,
        "max": 5,
        "step": 1,
    },
    "probThresh": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "Probability threshold",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "mutationProb": {
        "type": "SliderFloat",
        "value": 0.01,
        "label": "Mutation probability",
        "min": 0.0,
        "max": 0.2,
        "step": 0.005,
    },
}


@sl.component
def LanguageNetworkComponent(model: LanguageModel):
    # Get the graph from the model
    G = model.languageNetwork

    # Create a fresh figure each render
    fig, ax = plt.subplots(figsize=(5, 4))

    if len(G.nodes) > 0:
        # Layout the graph (spring layout works nicely for small graphs)
        pos = nx.spring_layout(G)

        # You can customize node colors based on origin if you store that as an attribute
        node_colors = "tab:blue"

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=300,
            node_color=node_colors,
            font_size=8,
            ax=ax,
        )

    ax.set_title("Language Network / Tree")
    ax.axis("off")

    # Wrap the figure as a Solara component
    return sl.FigureMatplotlib(fig)




language_model = LanguageModel(
    width=20,
    height=20,
    numAgents=100,
    P=None,
    r=1,
    probThresh=0.5,
    mutationProb=0.01,
)

renderer = SpaceRenderer(model=language_model, backend="matplotlib").render(
    agent_portrayal=agent_portrayal
)

# no page argument = page 0
mut_and_lang_plot = make_plot_component(
    ["num_languages", "num_mutations"]
)

# no tuple, just the component = page 0
language_network_component = LanguageNetworkComponent

viz = SolaraViz(
    language_model,
    renderer,
    components=[
        mut_and_lang_plot,
        language_network_component,
    ],
    model_params=model_params,
    name="Language Evolution Model",
)



@sl.component
def Page():
    return viz



