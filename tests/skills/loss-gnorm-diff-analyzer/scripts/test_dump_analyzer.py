# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

#!/usr/bin/env python3
"""dump_analyzer.py 单元测试：基础纯函数（解析/差异/归约/族名）+ 决定性锚点（守恒/身份/覆盖缺口/聚合悬殊/反向产生）+ 变体匹配/分歧普查。"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "skills",
        "accuracy",
        "loss-gnorm-diff-analyzer",
        "scripts",
    ),
)

from dump_analyzer import (
    _output_quad_index,
    _reduce_factor,
    agg_single_sided_check,
    backward_created_check,
    classify_pair_diff,
    conservation_check,
    coverage_gap_check,
    divergence_census,
    expand_node_slots,
    match_name_variants,
    norm_family,
    operand_identity_check,
    parse_node_name,
    tensor_pair_diff,
)


def _slot(dtype="torch.float32", norm=1.0):
    return {
        "role": "output",
        "path": "output.0",
        "leaf": "tensor",
        "dtype": dtype,
        "shape": [2, 3],
        "requires_grad": True,
        "Max": norm,
        "Min": -norm,
        "Mean": 0.0,
        "Norm": norm,
    }


def _tensor(dtype="torch.float32", shape=(2, 3), max=1.0, min=-1.0, mean=0.5, norm=1.0):
    """构造 dump.json 张量槽位元素（统计量 dict，键名与 dump.json 一致）。"""
    return {
        "type": "torch.Tensor",
        "dtype": dtype,
        "shape": list(shape),
        "Max": max,
        "Min": min,
        "Mean": mean,
        "Norm": norm,
        "requires_grad": True,
    }


def _node(name, inputs=None, outputs=None, order=0):
    """按 parse_side_dump 的节点记录结构构造节点（走真实 parse_node_name/expand_node_slots）。"""
    info = parse_node_name(name)
    data = {}
    if inputs is not None:
        data["input"] = inputs
    if outputs is not None:
        data["output"] = outputs
    return {
        "name": name,
        "order": order,
        "kind": info["kind"],
        "family": info["family"],
        "idx": info["idx"],
        "dir": info["dir"],
        "slots": expand_node_slots(data),
    }


def _reduce_bwd_node(name, mean_in, mean_out):
    """归约类反向节点：input [8,3] → output [4,3]（N=2），守恒 ⇔ mean(out) = 2*mean(in)。"""
    return _node(
        name,
        inputs={"0": _tensor(shape=(8, 3), mean=mean_in, norm=4.0)},
        outputs={"0": _tensor(shape=(4, 3), mean=mean_out, norm=2.0)},
    )


def _pg_rec():
    """参数梯度差异记录（param_grad 列表元素）。"""
    return {"node": "Module.enc.0.parameters_grad.0", "param": "output.0", "norm_t": 3.0, "norm_g": 1.0}


class TestParseNodeName(unittest.TestCase):
    def test_api_module_paramgrad(self):
        api = parse_node_name("Tensor.permute.0.forward")
        self.assertEqual((api["kind"], api["family"], api["dir"]), ("api", "Tensor.permute", "forward"))
        mod = parse_node_name("Module.time_embedding.0.Linear.forward.0")
        self.assertEqual((mod["kind"], mod["dir"]), ("module", "forward"))
        pg = parse_node_name("Module.proj_out.Linear.parameters_grad.0")
        self.assertEqual(pg["kind"], "paramgrad")

    def test_other(self):
        self.assertEqual(parse_node_name("weird.node")["kind"], "other")


class TestTensorPairDiff(unittest.TestCase):
    def test_identical_and_significant(self):
        d0 = tensor_pair_diff(_slot(norm=2.0), _slot(norm=2.0))
        self.assertEqual(d0["nre"], 0.0)
        self.assertEqual(classify_pair_diff(d0), "none")
        d1 = tensor_pair_diff(_slot(norm=1.0), _slot(norm=1.1))
        self.assertAlmostEqual(d1["nre"], 0.1 / 1.1)
        self.assertEqual(classify_pair_diff(d1), "significant")

    def test_noise_and_one_side_zero(self):
        self.assertTrue(tensor_pair_diff(_slot(norm=1e-9), _slot(norm=2e-9))["noise"])
        self.assertTrue(tensor_pair_diff(_slot(norm=0.0), _slot(norm=1.0))["one_side_zero"])

    def test_structural(self):
        d = tensor_pair_diff(_slot(dtype="torch.float16"), _slot())
        self.assertTrue(d["dtype_diff"])
        self.assertEqual(classify_pair_diff(d), "structural")


class TestReduceAndFamily(unittest.TestCase):
    def test_reduce_factor(self):
        self.assertEqual(_reduce_factor([8, 3], [4, 3]), (2, 0))
        self.assertEqual(_reduce_factor([6, 3], [3]), (6, 0))
        self.assertIsNone(_reduce_factor([2, 3], [3, 2]))

    def test_norm_family(self):
        self.assertEqual(norm_family("Tensor.permute.forward_rank3"), "Tensor.permute.forward")
        self.assertEqual(norm_family("Tensor.permute.forward"), "Tensor.permute.forward")


class TestMatchNameVariants(unittest.TestCase):
    def test_rank_variant_paired_with_shape_overlap(self):
        # 族名归一化（剥离 _rank<N>）后 (族, 调用号, 方向) 双侧唯一 + 形状有交集 → 1:1 配对
        t_name, g_name = "Tensor.add_rank0.0.forward", "Tensor.add.0.forward"
        t_nodes = {t_name: _node(t_name, inputs={"0": _tensor()}, outputs={"0": _tensor()})}
        g_nodes = {g_name: _node(g_name, inputs={"0": _tensor()}, outputs={"0": _tensor()})}
        self.assertEqual(match_name_variants([t_name], [g_name], t_nodes, g_nodes), {t_name: g_name})

    def test_shape_gate_blocks_pairing(self):
        # 两侧张量槽位形状无交集（防分片/不可比配对）→ 放弃配对
        t_name, g_name = "Tensor.add_rank0.0.forward", "Tensor.add.0.forward"
        t_nodes = {t_name: _node(t_name, inputs={"0": _tensor(shape=(2, 3))})}
        g_nodes = {g_name: _node(g_name, inputs={"0": _tensor(shape=(4, 5))})}
        self.assertEqual(match_name_variants([t_name], [g_name], t_nodes, g_nodes), {})

    def test_ambiguous_key_not_paired(self):
        # 归一化后同键在 target 侧出现两次 → 不满足双侧唯一，放弃 1:1 配对
        t_names = ["Tensor.add_rank0.0.forward", "Tensor.add_rank1.0.forward"]
        g_name = "Tensor.add.0.forward"
        t_nodes = {n: _node(n, inputs={"0": _tensor()}) for n in t_names}
        g_nodes = {g_name: _node(g_name, inputs={"0": _tensor()})}
        self.assertEqual(match_name_variants(t_names, [g_name], t_nodes, g_nodes), {})


class TestDivergenceCensus(unittest.TestCase):
    def test_forward_buckets_and_backward_empty(self):
        # 同形状 output 槽位按 nre 分桶计数；单侧地板外 / 比值≥1e3 / 非有限独立计数
        name = "Tensor.cat.0.forward"
        outs_t = {
            "0": _tensor(max=2.0, min=-2.0, mean=0.3, norm=2.0),
            "1": _tensor(max=1.0, min=-1.0, mean=0.2, norm=1.0),
            "2": _tensor(max=0.0, min=0.0, mean=0.0, norm=0.0),
            "3": _tensor(max=2.0, min=-2.0, mean=0.3, norm=2.0),
            "4": _tensor(max=float("nan"), min=-1.0, mean=0.2, norm=float("nan")),
        }
        outs_g = {
            "0": _tensor(max=2.0, min=-2.0, mean=0.3, norm=2.0),  # nre=0 → rel0
            "1": _tensor(max=2.0, min=-2.0, mean=0.3, norm=2.0),  # nre=0.5 → significant
            "2": _tensor(max=1.0, min=-1.0, mean=0.2, norm=1.0),  # 单侧地板外+比值≥1e3+nre=1
            "3": _tensor(max=2.0, min=-2.0, mean=0.3, norm=2.001),  # nre<1e-3 → minor
            "4": _tensor(max=1.0, min=-1.0, mean=0.2, norm=1.0),  # 非有限（target 侧 NaN）
        }
        t_nodes = {name: _node(name, outputs=outs_t)}
        g_nodes = {name: _node(name, outputs=outs_g)}
        res = divergence_census([(name, name, "exact")], t_nodes, g_nodes)
        fwd = res["forward"]
        self.assertEqual(fwd["same_shape_slots"], 5)
        self.assertEqual(fwd["rel0"], 1)
        self.assertEqual(fwd["minor"], 1)
        self.assertEqual(fwd["significant"], 2)
        self.assertEqual(fwd["one_side_zero"], 1)
        self.assertEqual(fwd["ratio_ge_1e3"], 1)
        self.assertEqual(fwd["nonfinite"], 1)
        self.assertEqual(fwd["rel_min"], 0.0)
        self.assertAlmostEqual(fwd["rel_max"], 1.0)
        self.assertEqual(fwd["top_examples"][0]["path"], "output.2")
        # 无反向配对节点 → backward 侧全零
        self.assertEqual(res["backward"]["same_shape_slots"], 0)
        self.assertIsNone(res["backward"]["rel_min"])


class TestConservationCheck(unittest.TestCase):
    def test_split_violation_flagged(self):
        # N=2：target 守恒（N_eff=2）、golden 违反（N_eff=1）→ flagged 且 cons_t != cons_g
        name = "Tensor.mean.0.backward"
        t = {name: _reduce_bwd_node(name, 1.0, 2.0)}
        g = {name: _reduce_bwd_node(name, 1.0, 1.0)}
        rows, flagged = conservation_check([(name, name, "exact")], t, g)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(flagged), 1)
        r = flagged[0]
        self.assertEqual(r["N"], 2)
        self.assertEqual(r["N_eff_t"], 2.0)
        self.assertEqual(r["N_eff_g"], 1.0)
        self.assertTrue(r["cons_t"])
        self.assertFalse(r["cons_g"])

    def test_both_conserve_not_flagged(self):
        name = "Tensor.mean.0.backward"
        t = {name: _reduce_bwd_node(name, 1.0, 2.0)}
        g = {name: _reduce_bwd_node(name, 1.0, 2.0)}
        rows, flagged = conservation_check([(name, name, "exact")], t, g)
        self.assertEqual(len(rows), 1)
        self.assertEqual(flagged, [])

    def test_both_violate_same_neff_not_flagged(self):
        # 两侧同样违反且 N_eff 相同 → mean 类反向的语义性"违反"，不属于分歧锚点
        name = "Tensor.mean.0.backward"
        t = {name: _reduce_bwd_node(name, 1.0, 1.0)}
        g = {name: _reduce_bwd_node(name, 1.0, 1.0)}
        _, flagged = conservation_check([(name, name, "exact")], t, g)
        self.assertEqual(flagged, [])


class TestOperandIdentityCheck(unittest.TestCase):
    @staticmethod
    def _nodes(cons_input_t, cons_input_g):
        producer = "Tensor.add.0.forward"
        consumer = "Tensor.mul.0.backward"
        t_nodes = {
            producer: _node(producer, outputs={"0": _tensor(max=3.0, min=-3.0, mean=0.5, norm=3.0)}),
            consumer: _node(
                consumer,
                inputs={"0": cons_input_t},
                outputs={"0": _tensor(max=9.0, min=-9.0, mean=1.5, norm=9.0)},
            ),
        }
        g_nodes = {
            producer: _node(producer, outputs={"0": _tensor(max=7.0, min=-7.0, mean=1.5, norm=7.0)}),
            consumer: _node(
                consumer,
                inputs={"0": cons_input_g},
                outputs={"0": _tensor(max=9.0, min=-9.0, mean=1.5, norm=9.0)},
            ),
        }
        return producer, consumer, t_nodes, g_nodes

    def test_target_only_hit(self):
        # target 操作数四元组命中本侧某节点输出、golden 无任何命中 → 命中存在性不对称（高置信）
        producer, consumer, t_nodes, g_nodes = self._nodes(
            _tensor(max=3.0, min=-3.0, mean=0.5, norm=3.0),
            _tensor(max=5.0, min=-5.0, mean=1.0, norm=5.0),
        )
        groups = operand_identity_check(
            [(consumer, consumer, "exact")],
            t_nodes,
            g_nodes,
            _output_quad_index(t_nodes),
            _output_quad_index(g_nodes),
        )
        self.assertEqual(len(groups), 1)
        grp = groups[0]
        self.assertEqual(grp["family"], "Tensor.mul")
        self.assertEqual(grp["path"], "input.0")
        self.assertEqual(grp["hits_side"], "target_only")
        self.assertEqual(grp["confidence"], "high")
        self.assertEqual(grp["count"], 1)
        ex = grp["examples"][0]
        self.assertEqual(ex["hits_target"], [producer])
        self.assertEqual(ex["hits_golden"], [])
        self.assertEqual(ex["quad_target"], [3.0, -3.0, 0.5, 3.0])

    def test_both_sides_hit_not_flagged(self):
        # golden 操作数同样命中本侧输出 → 命中存在性对称 → 无组
        _producer, consumer, t_nodes, g_nodes = self._nodes(
            _tensor(max=3.0, min=-3.0, mean=0.5, norm=3.0),
            _tensor(max=7.0, min=-7.0, mean=1.5, norm=7.0),
        )
        groups = operand_identity_check(
            [(consumer, consumer, "exact")],
            t_nodes,
            g_nodes,
            _output_quad_index(t_nodes),
            _output_quad_index(g_nodes),
        )
        self.assertEqual(groups, [])

    def test_variant_match_lowers_confidence(self):
        # 变体匹配对（非 exact）产生同样的命中不对称，但置信度降为 low
        t_name, g_name = "Tensor.mul_rank0.0.backward", "Tensor.mul.0.backward"
        t_nodes = {
            "Tensor.add.0.forward": _node(
                "Tensor.add.0.forward", outputs={"0": _tensor(max=3.0, min=-3.0, mean=0.5, norm=3.0)}
            ),
            t_name: _node(t_name, inputs={"0": _tensor(max=3.0, min=-3.0, mean=0.5, norm=3.0)}),
        }
        g_nodes = {g_name: _node(g_name, inputs={"0": _tensor(max=5.0, min=-5.0, mean=1.0, norm=5.0)})}
        groups = operand_identity_check(
            [(t_name, g_name, "name_variant")],
            t_nodes,
            g_nodes,
            _output_quad_index(t_nodes),
            _output_quad_index(g_nodes),
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["confidence"], "low")
        self.assertEqual(groups[0]["n_variant"], 1)


class TestCoverageGapCheck(unittest.TestCase):
    def test_backward_uncaptured_on_target(self):
        # 参数梯度所属模块的反向节点不在 target 捕获窗口 → 覆盖缺口（分歧产生于未捕获反向段）
        fwd_name = "Module.enc.0.Linear.forward.0"
        t_nodes = {fwd_name: _node(fwd_name, outputs={"0": _tensor()})}
        g_nodes = {"Module.enc.0.backward.0": _node("Module.enc.0.backward.0", inputs={"0": _tensor()})}
        rows = coverage_gap_check([], t_nodes, g_nodes, [_pg_rec()])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["module"], "Module.enc.0")
        self.assertFalse(r["bwd_captured_t"])
        self.assertTrue(r["bwd_captured_g"])
        self.assertFalse(r["bwd_matched"])

    def test_backward_captured_and_matched(self):
        # 模块反向两侧捕获且配对 → 无覆盖缺口
        name = "Module.enc.0.backward.0"
        t_nodes = {name: _node(name, inputs={"0": _tensor()})}
        g_nodes = {name: _node(name, inputs={"0": _tensor()})}
        rows = coverage_gap_check([(name, name, "exact")], t_nodes, g_nodes, [_pg_rec()])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertTrue(r["bwd_captured_t"])
        self.assertTrue(r["bwd_captured_g"])
        self.assertTrue(r["bwd_matched"])


class TestAggSingleSidedCheck(unittest.TestCase):
    NAME = "Tensor._foreach_norm.0.backward"

    def _pair(self, in_t, in_g):
        t_nodes = {self.NAME: _node(self.NAME, inputs=in_t)}
        g_nodes = {self.NAME: _node(self.NAME, inputs=in_g)}
        return [(self.NAME, self.NAME, "exact")], t_nodes, g_nodes

    @staticmethod
    def _skewed_inputs():
        # 2 个输入槽位 target 侧 Norm 悬殊 ≥100×（聚合输入 = .grad 本体）
        return (
            {
                "0": _tensor(max=200.0, min=-200.0, mean=20.0, norm=200.0),
                "1": _tensor(max=300.0, min=-300.0, mean=30.0, norm=300.0),
            },
            {
                "0": _tensor(max=1.0, min=-1.0, mean=0.1, norm=1.0),
                "1": _tensor(max=1.0, min=-1.0, mean=0.1, norm=1.0),
            },
        )

    def test_window_out_without_producer(self):
        # 悬殊输入四元组无同侧输出命中（占多数）→ window_out 锚点
        pairs, t_nodes, g_nodes = self._pair(*self._skewed_inputs())
        flags = agg_single_sided_check(pairs, t_nodes, g_nodes, {}, {})
        self.assertEqual(len(flags), 1)
        f = flags[0]
        self.assertEqual(f["node"], self.NAME)
        self.assertEqual(f["side"], "target")
        self.assertEqual(f["n_flagged"], 2)
        self.assertEqual(f["n_no_producer"], 2)
        self.assertTrue(f["window_out"])
        self.assertAlmostEqual(f["top_ratio"], 300.0)

    def test_producer_hit_clears_window_out(self):
        # 悬殊输入四元组全部命中同侧输出（窗口内有生产者）→ 仍 flag 但 window_out=False
        producer = _node(
            "Tensor.mul.0.forward",
            outputs={
                "0": _tensor(max=200.0, min=-200.0, mean=20.0, norm=200.0),
                "1": _tensor(max=300.0, min=-300.0, mean=30.0, norm=300.0),
            },
        )
        idx_t = _output_quad_index({"Tensor.mul.0.forward": producer})
        pairs, t_nodes, g_nodes = self._pair(*self._skewed_inputs())
        flags = agg_single_sided_check(pairs, t_nodes, g_nodes, idx_t, {})
        self.assertEqual(len(flags), 1)
        f = flags[0]
        self.assertFalse(f["window_out"])
        self.assertEqual(f["n_no_producer"], 0)
        for row in f["rows"]:
            self.assertEqual(row["producers"], ["Tensor.mul.0.forward"])

    def test_below_ratio_or_single_slot_not_flagged(self):
        # 比值未达 ratio_k=100，或悬殊槽位仅 1 个（<2）→ 不成锚点
        pairs, t_nodes, g_nodes = self._pair(
            {"0": _tensor(max=50.0, min=-50.0, mean=5.0, norm=50.0)},
            {"0": _tensor(max=1.0, min=-1.0, mean=0.1, norm=1.0)},
        )
        self.assertEqual(agg_single_sided_check(pairs, t_nodes, g_nodes, {}, {}), [])
        pairs, t_nodes, g_nodes = self._pair(
            {"0": _tensor(max=200.0, min=-200.0, mean=20.0, norm=200.0)},
            {"0": _tensor(max=1.0, min=-1.0, mean=0.1, norm=1.0)},
        )
        self.assertEqual(agg_single_sided_check(pairs, t_nodes, g_nodes, {}, {}), [])


class TestBackwardCreatedCheck(unittest.TestCase):
    def test_input_identical_output_divergent(self):
        # 输入四元组双侧逐位相等 + 输出 NRE≥1e-3 → 反向内产生分歧锚点（非继承）
        name = "Tensor.add.0.backward"
        t = _node(
            name,
            inputs={"0": _tensor(max=2.0, min=-2.0, mean=0.4, norm=2.0)},
            outputs={"0": _tensor(norm=1.0)},
        )
        g = _node(
            name,
            inputs={"0": _tensor(max=2.0, min=-2.0, mean=0.4, norm=2.0)},
            outputs={"0": _tensor(norm=2.0)},
        )
        out = backward_created_check([(name, name, "exact")], {name: t}, {name: g})
        self.assertEqual(len(out), 1)
        r = out[0]
        self.assertEqual(r["n_input_slots"], 1)
        self.assertEqual(r["diverge_path"], "output.0")
        self.assertAlmostEqual(r["nre"], 0.5)
        self.assertEqual(r["norm_t"], 1.0)
        self.assertEqual(r["norm_g"], 2.0)

    def test_divergent_input_not_flagged(self):
        # 任一输入槽位四元组不一致 → 不可判"分歧在该反向内部产生"，跳过
        name = "Tensor.add.0.backward"
        t = _node(
            name,
            inputs={"0": _tensor(max=2.0, min=-2.0, mean=0.4, norm=2.0)},
            outputs={"0": _tensor(norm=1.0)},
        )
        g = _node(
            name,
            inputs={"0": _tensor(max=2.0, min=-2.0, mean=0.4, norm=2.1)},
            outputs={"0": _tensor(norm=2.0)},
        )
        self.assertEqual(backward_created_check([(name, name, "exact")], {name: t}, {name: g}), [])

    def test_output_below_threshold_not_flagged(self):
        # 输入一致但输出 NRE < DIRTY_NRE(1e-3) → 不构成锚点
        name = "Tensor.add.0.backward"
        t = _node(
            name,
            inputs={"0": _tensor(max=2.0, min=-2.0, mean=0.4, norm=2.0)},
            outputs={"0": _tensor(norm=1.0)},
        )
        g = _node(
            name,
            inputs={"0": _tensor(max=2.0, min=-2.0, mean=0.4, norm=2.0)},
            outputs={"0": _tensor(norm=1.0005)},
        )
        self.assertEqual(backward_created_check([(name, name, "exact")], {name: t}, {name: g}), [])


if __name__ == "__main__":
    unittest.main()
