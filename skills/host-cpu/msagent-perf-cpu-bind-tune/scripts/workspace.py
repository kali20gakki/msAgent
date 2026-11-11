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

"""工作空间与输出目录解析。

约定：
  工作空间 = Agent 运行时设置的工作路径（cwd），由宿主 / 远程执行插件决定；
  skill 不猜测自身安装位置，也不自行选择远端目录（远端拷贝与 cwd 由远程执行侧控制）。
  运行产物统一落 <workspace>/workspaces/<host>/<timestamp>/。

优先级：显式 root 参数 > HOSTCPU_WORKSPACE > 当前工作目录（cwd）。
"""

import os
import re
import socket
import time


def safe_name(value):
    """主机名等外部输入进路径前的清洗：仅保留字母数字/点/横线/下划线。

    socket.gethostname() 在个别环境可能返回含路径分隔符或超长字符串，
    统一替换并截断，防止产物落盘位置被意外改写。
    """
    cleaned = re.sub(r"[^\w.-]+", "_", str(value or "")).strip("._")
    # 截断后再 rstrip：避免第 64 个字符恰为 . / _ 时留下非法尾字符
    return (cleaned or "unknown")[:64].rstrip("._") or "unknown"


def workspace_root(root=None):
    if root:
        return os.path.abspath(os.path.expanduser(root))
    env = os.environ.get("HOSTCPU_WORKSPACE")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.abspath(os.getcwd())


def session_dir(host=None, timestamp=None, root=None, create=True):
    """计算（并默认创建）本次会话的产物目录。

    create=False 时只返回路径、不落盘——供 --dry-run 使用，
    保证 dry-run 语义为「不产生任何文件系统副作用」。
    """
    host = safe_name(host or socket.gethostname())
    timestamp = safe_name(timestamp or time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    path = os.path.join(workspace_root(root), "workspaces", host, timestamp)
    if create:
        os.makedirs(path, exist_ok=True)
    return path
