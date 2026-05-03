from cote_paper.replay import ReplayStep, TrajectoryReplay


def test_replay_records_edge_messages_and_actions():
    replay = TrajectoryReplay()
    replay.add_step(ReplayStep(step=0, context={"myPos": 1}, action=2, edge_messages={"T1->T8": "x"}))
    assert replay.steps[0].action == 2
    assert replay.steps[0].edge_messages["T1->T8"] == "x"


def test_replay_to_json_is_stable():
    replay = TrajectoryReplay()
    replay.add_step(ReplayStep(step=1, context={"stage": "play"}, action=3, action_distribution=[0.2, 0.8]))
    payload = replay.to_json()
    assert payload["steps"][0]["step"] == 1
    assert payload["steps"][0]["action_distribution"] == [0.2, 0.8]
