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

"""线程角色规则唯一来源：comm 关键字 → comm / io / compute / main。

discover（建模）、apply_plan（执行）、verify_config（校验）共用本模块，
避免三处各自维护一份关键字导致口径漂移。
"""

ROLE_ORDER = ("comm", "io", "compute", "main")

ROLE_PATTERNS = (
    ("comm", ("hccl", "aoe_rtkb")),
    ("io", ("zmq", "tokenizer", "detokenizer", "dataloader", "network")),
    ("compute", ("vllm", "caffe", "python", "pt_autograd", "acl_thread", "compl", "omp")),
)


def classify(comm):
    c = (comm or "").lower()
    for role, keys in ROLE_PATTERNS:
        if any(k in c for k in keys):
            return role
    return "main"


def keywords(role):
    for name, keys in ROLE_PATTERNS:
        if name == role:
            return list(keys)
    return []
