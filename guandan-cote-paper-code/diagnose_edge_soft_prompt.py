# -*- coding: utf-8 -*-
"""
诊断脚本：验证 edge message 生成是否因 soft_prompt_vector 注入而失败

对比：
  A. 不带 soft_prompt_vector → generate_text(..., soft_prompt_vector=None)
  B. 带 soft_prompt_vector   → generate_text(..., soft_prompt_vector=vector)

预期结果：
  A 成功 & B 失败  → 问题在 soft prompt 注入路径
  A 成功 & B 成功  → 说明 generate_text 本身没问题，问题在别处
  A 失败           → 模型或 generate_text 有其他问题
"""
from __future__ import annotations
import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from cote_paper.local_model import LocalCausalLM
from cote_paper.game import GameContext


def make_fake_context() -> GameContext:
    return GameContext(
        my_pos=1,
        partner_pos=3,
        opponents=(0, 2),
        hand_cards=["SA", "HA", "CA", "DA", "SK"],
        public_info=[{}, {}, {}, {}],   # 每个座位一个 dict，min_opponent_rest 需要这个
        self_rank="2",
        oppo_rank="2",
        cur_rank="2",
        cur_pos=1,
        cur_action=None,
        greater_pos=None,
        greater_action=None,
        stage="play",
    )


def main() -> int:
    print("=" * 60)
    print("Edge Message Soft Prompt 诊断")
    print("=" * 60)

    lm = LocalCausalLM()
    ok = lm.ensure_loaded()
    print(f"\n模型加载: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"  last_error: {lm.last_error}")
        return 1
    print(f"  hidden_size: {lm.hidden_size}")

    edge = ("T1_board_parser", "T2_history_tracker")
    # 生成一个合法维度的 vector（等于 hidden_size）
    vector = [0.05 * (i % 10 - 5) for i in range(lm.hidden_size or 64)]
    source_belief = [0.125] * 8
    context = make_fake_context()

    print(f"\nEdge: {edge}")
    print(f"vector 维度: {len(vector)}")

    # 构造与 generate_edge_message 相同的 system + payload
    from cote_paper.prompts import decode_soft_prompt
    prompt_text = decode_soft_prompt(vector, edge, compact=False)
    system = (
        "You are an internal COTE communication node. "
        "Return one compact JSON micro-message. "
        "Keys must come from finish, block_opponent, help_partner, preserve_bomb, shed_cards, low_ambiguity."
    )
    payload = json.dumps({
        "edge": f"{edge[0]}->{edge[1]}",
        "softPromptProjection": prompt_text,
        "sourceBelief": [round(v, 4) for v in source_belief],
        "state": lm._state_payload(context),
    }, ensure_ascii=False, separators=(",", ":"))

    # ---- 测试 A：无 soft_prompt_vector ----
    print("\n--- 测试 A：不带 soft_prompt_vector ---")
    lm.last_error = None
    raw_a = lm.generate_text(system, payload, max_new_tokens=128, soft_prompt_vector=None)
    parsed_a = lm.generate_json(system, payload, max_new_tokens=128, soft_prompt_vector=None)
    print(f"  generate_text: {repr(raw_a)[:150] if raw_a else 'None'}")
    print(f"  generate_json: {parsed_a if parsed_a else 'None'}")
    print(f"  error: {lm.last_error}")

    # ---- 测试 B：有 soft_prompt_vector ----
    print("\n--- 测试 B：带 soft_prompt_vector（inputs_embeds 注入）---")
    lm.last_error = None
    raw_b = lm.generate_text(system, payload, max_new_tokens=128, soft_prompt_vector=vector)
    parsed_b = lm.generate_json(system, payload, max_new_tokens=128, soft_prompt_vector=vector)
    print(f"  generate_text: {repr(raw_b)[:150] if raw_b else 'None'}")
    print(f"  generate_json: {parsed_b if parsed_b else 'None'}")
    print(f"  error: {lm.last_error}")

    # ---- 测试 C：generate_edge_message（当前实现）----
    print("\n--- 测试 C：generate_edge_message（完整包装）---")
    lm.last_error = None
    result_c = lm.generate_edge_message(edge=edge, vector=vector,
                                        source_belief=source_belief, context=context)
    print(f"  结果: {result_c if result_c else 'None'}")
    print(f"  error: {lm.last_error}")

    # ---- 结论 ----
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)
    success_a = parsed_a is not None
    success_b = parsed_b is not None
    success_c = result_c is not None

    print(f"  A（无 soft_prompt）: {'成功' if success_a else '失败'}")
    print(f"  B（有 soft_prompt） : {'成功' if success_b else '失败'}")
    print(f"  C（generate_edge_message）: {'成功' if success_c else '失败'}")

    if success_a and not success_b:
        print("\n  → soft_prompt_vector 注入路径有问题，B 失败而 A 成功")
        print("    这说明 inputs_embeds 分支的 offset 修正不完整，或模型对 soft prefix 响应异常。")
    elif success_a and success_b and not success_c:
        print("\n  → generate_text 带/不带 soft 都成功，但 generate_edge_message 失败")
        print("    问题在 generate_edge_message 的包装层（payload 构造、decode_soft_prompt 等）")
    elif success_a and success_b:
        print("\n  → generate_text 两个分支都正常，edge 问题与 soft_prompt_vector 无关")
        print("    检查 generate_edge_message 的其他逻辑")
    else:
        print("\n  → generate_text 两个分支都失败，说明模型加载或 generate_text 有其他问题")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
