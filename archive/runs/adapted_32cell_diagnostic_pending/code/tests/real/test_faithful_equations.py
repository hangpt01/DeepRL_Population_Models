import numpy as np

from real_ecology_benchmark.faithful_ecology import default_model


def test_registered_equations_match_hand_values():
    x = 0.6
    for form in ("ricker", "allee", "theta", "regime"):
        model = default_model(form, 2, 100.0, 0.1)
        actual = float(model.noiseless_next(x, 1.0, 0, 0))
        if form == "ricker":
            expected = x * np.exp(0.2 * (1.0 - x))
        elif form == "allee":
            expected = x * np.exp(0.2 * (1.0 - x) * (x / 0.2 - 1.0))
        elif form == "theta":
            expected = max(0.0, x + 0.2 * x * (1.0 - x ** 2.0))
        else:
            expected = x * np.exp(0.7 * 0.2 * (1.0 - x) * (x / 0.2 - 1.0))
        assert np.isclose(actual, expected)


def test_zero_is_absorbing_unless_stocking_is_fitted():
    model = default_model("ricker", 2, 100.0, 0.1)
    assert float(model.noiseless_next(0.0, 1.0, 0)) == 0.0
    stocking = model.stocking.copy()
    stocking[1] = 0.1
    channels = list(model.action_channels)
    channels[1] = "state"
    stocked = type(model)(**{
        **model.__dict__, "stocking": stocking, "action_channels": tuple(channels)
    })
    assert float(stocked.noiseless_next(0.0, 1.0, 1)) > 0.0


def test_capacity_update_is_deterministic_and_bounded():
    model = default_model("ricker", 2, 100.0, 0.1)
    increments = model.capacity_increment.copy()
    increments[1] = 0.4
    channels = list(model.action_channels)
    channels[1] = "capacity"
    model = type(model)(**{
        **model.__dict__, "capacity_increment": increments,
        "action_channels": tuple(channels),
    })
    assert model.next_capacity(1.0, 1) == 1.4
    assert model.next_capacity(1.9, 1) == model.capacity_ceiling
