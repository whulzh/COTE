from cote_paper.local_model import repair_node_thought_json


def test_repairs_act_index_colon_score_pairs():
    text = (
        '{"node":"T7_hand_value","summary":"Evaluating hand",'
        '"candidateScores":{"actIndex":0:0.85,"actIndex":1:-0.95},"confidence":0.9}'
    )

    parsed = repair_node_thought_json(text, "T7_hand_value")

    assert parsed is not None
    assert parsed["node"] == "T7_hand_value"
    assert parsed["candidateScores"] == {"0": 0.85, "1": -0.95}
    assert parsed["confidence"] == 0.9


def test_repairs_act_index_comma_score_pair():
    text = (
        '{"node":"T7_hand_value","summary":"Evaluating hand",'
        '"candidateScores":{"actIndex":75,0.85},"confidence":0.9}'
    )

    parsed = repair_node_thought_json(text, "T7_hand_value")

    assert parsed is not None
    assert parsed["candidateScores"] == {"75": 0.85}


def test_repair_returns_none_without_scores():
    assert repair_node_thought_json('{"summary":"no scores"}', "T1_board_parser") is None
