import numpy as np

from modelv2 import LanguageModel, LanguageAgent


def make_model():
    # Small, simple model for tests
    return LanguageModel(
        width=10,
        height=10,
        numAgents=20,
        P=None,
        r=1,
        probThresh=0.1,
        mutationProb=0.1,
        languageNum=3,
    )


def test_model_init_basic():
    m = make_model()
    # Grid exists and dimensions match
    assert m.grid is not None
    assert m.grid.width == m.width == 10
    assert m.grid.height == m.height == 10

    # P is square and matches languageNum
    assert m.P.shape[0] == m.P.shape[1]
    assert m.P.shape[0] == m.languageNum

    # Probabilities in [0,1]
    assert np.all(m.P >= 0) and np.all(m.P <= 1)

    # Language network created with all base languages as nodes
    assert m.languageNetwork is not None
    for lang in range(m.languageNum):
        assert lang in m.languageNetwork.nodes


def test_agents_created_and_on_grid():
    m = make_model()

    # Mesa 3: model.agents is an AgentSet
    agents = list(m.language_agents)
    assert len(agents) == m.numAgents

    for a in agents:
        # Type check
        assert isinstance(a, LanguageAgent)

        # Position is within bounds
        x, y = a.pos
        assert 0 <= x < m.width
        assert 0 <= y < m.height

        # Language index is valid
        assert 0 <= a.language < m.languageNum


def test_agent_moves_stay_in_bounds():
    m = make_model()
    agents_before = {id(a): a.pos for a in m.language_agents}

    # One step of the model
    m.step()

    for a in m.language_agents:
        x, y = a.pos
        # After move, still within bounds
        assert 0 <= x < m.width
        assert 0 <= y < m.height


def test_mutation_increases_language_space():
    m = make_model()

    # Force a mutation scenario: pick two agents and call createLanguageMutuation
    agents = list(m.language_agents)
    a1, a2 = agents[0], agents[1]

    old_lang_num = m.languageNum
    old_P_shape = m.P.shape

    new_id = m.createLanguageMutuation(a1, a2)

    # languageNum increased by 1
    assert m.languageNum == old_lang_num + 1

    # P expanded by 1 in each dimension
    assert m.P.shape[0] == old_P_shape[0] + 1
    assert m.P.shape[1] == old_P_shape[1] + 1

    # New language id is last index
    assert new_id == old_lang_num
    assert 0 <= new_id < m.languageNum

    # Network updated and contains new node
    assert new_id in m.languageNetwork.nodes


if __name__ == "__main__":
    # Simple runner without pytest
    tests = [
        test_model_init_basic,
        test_agents_created_and_on_grid,
        test_agent_moves_stay_in_bounds,
        test_mutation_increases_language_space,
    ]

    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print(f"{test.__name__} passed ✅")

    print("All tests passed.")
