"""Declare the prior supplied to the existing informed fitting controls."""

from hidden_process_bench.config import Shape


def informed_scope(shape: Shape) -> dict[str, object]:
    """Checkpoint qualification and fitting accuracy do not establish discovery."""
    return {
        "track": "informed_hmm_parameter_fit",
        "qualification_scope": "transformer_prediction_only",
        "disclosed_process_family": "mixture_of_node_emitting_hmms",
        "disclosed_components": shape.components,
        "disclosed_states_per_component": shape.states,
        "family_discovery_evaluated": False,
        "independent_structure_recovery_evaluated": False,
    }
