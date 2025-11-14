import solara as sl
from mesa.visualization import SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle

from modelv2 import LanguageModel


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

def agent_portrayal(agent):
    # map language index -> color
    idx = int(agent.language) % len(LANG_COLORS)
    return AgentPortrayalStyle(
        color=LANG_COLORS[idx],
        marker="o",
        size=40,
    )


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

viz = SolaraViz(
    language_model,
    renderer,
    components=[],
    model_params=model_params,
    name="Language Evolution Model",
)


@sl.component
def Page():
    return viz



