# Best Practices for `torch_npu.profiler` Interface Adaptation

## 1. Adapting the `torch_npu.profiler` Interface for the `sglang` Framework in Inference Scenarios

Taking the `sglang` framework in inference scenarios as an example, this section shows how to complete `torch_npu.profiler` interface adaptation end-to-end using the `ascend-profiler-collect-adaption` skill of msagent.

### 1.1 Process

1. You provide the code of the `sglang` framework and state that `torch_npu.profiler` needs to be integrated.
2. msagent calls the `ascend-profiler-collect-adaption` skill to identify the actual startup entry point, the parameter parsing location, and the profiler injection point in inference scenarios.
3. Without changing the existing service logic, the skill adds the `torch_npu.profiler` related parameters, the configuration that is disabled by default, and the collection invocation logic.
4. In the verification and output stage, the skill checks whether the profiler parameters have been passed through, and confirms that the entire chain from external parameters to the core collection code works.
5. The skill verifies that the profiler data is complete, confirms that profiling data has been collected properly, and confirms that real NPU operators have been collected on the `timeline`.

### 1.2 Demonstration

![Inference scenario sglang framework interface adaptation process](../figures/best_practices/sglang_torch_npu_profiler.gif)

---

## 2. Adapting the `torch_npu.profiler` Interface for the `slime-ascend` Framework in Reinforcement Learning Scenarios

Taking the `slime-ascend` framework in reinforcement learning scenarios as an example, this section shows how to complete `torch_npu.profiler` interface adaptation end-to-end using the `ascend-profiler-collect-adaption` skill of msagent.

### 2.1 Process

1. You provide the code of the `slime-ascend` framework in reinforcement learning scenarios and state that `torch_npu.profiler` needs to be integrated.
2. msagent calls the `ascend-profiler-collect-adaption` skill to locate the injection position of `torch_npu.profiler` in reinforcement learning scenarios.
3. If the reinforcement learning framework contains rollout and actor stages, the skill adapts the `torch_npu.profiler` interface to the rollout and actor stages separately, and controls collection independently by stage to avoid cross-stage interference.
4. Without changing the existing service logic, the skill adds the `torch_npu.profiler` invocation, parameter pass-through, and configuration parsing logic.
5. The skill verifies that the profiler data is complete, confirms that profiling data has been collected properly, and confirms that real NPU operators have been collected on the `timeline` in both the actor and rollout stages.

### 2.2 Demonstration

![Training scenario slime-ascend framework interface adaptation process](../figures/best_practices/slime_ascend_torch_npu_profiler.gif)
