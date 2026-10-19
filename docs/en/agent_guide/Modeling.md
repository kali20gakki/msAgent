# Modeling: Simulation Modeling

`Modeling` is a domain agent for the msModeling scenario. It handles LLM/VLM simulation modeling problems, including environment initialization, performance modeling, single-point simulation, throughput planning, device profiles, and serving auto-optimization.

> The current version has integrated the msModeling dedicated skills (including the serving auto-optimization chain). Through the unified entry point, it can handle parameter completion, command planning, device modeling guidance, deployment and installation of the serving auto-optimization tool, optimization parameter recommendations, and `config.toml` configuration. When real performance results are involved, still rely on the output of actual runs.

## 1. Agent Positioning

- Targets the msModeling repository and its related tasks.
- Covers the following major directions: single-point performance simulation of LLMs and deployment throughput planning.
- Focuses on the following chain: understand requirements, organize inputs, form candidate commands or plans, confirm before execution, and summarize results or provide a verification path.
- Handles only the chains covered by the integrated dedicated skills. Capabilities that are not integrated fall outside the current scope.

## 2. Integrated Capabilities

The following msModeling dedicated capabilities are currently integrated:

- `msmodeling-env-installer`: Installs and verifies the msModeling development environment, covering creating `myenv` with `uv`, installing `requirements.txt`, configuring `PYTHONPATH` and `HF_ENDPOINT` for the current session, and checking dependency consistency.
- `msmodeling-text-generate-executor`: Completes parameters for single-point simulation, generates candidate commands, confirms before execution, and summarizes results.
- `msmodeling-throughput-optimizer-executor`: Completes parameters and summarizes results for throughput planning, hardware comparison, and aggregation, separation, and P:D ratio search.
- `msmodeling-device-config`: Converts natural language hardware specifications into device profile modeling input and explicitly marks the items to be calibrated.
- `msmodeling-optix-deploy`: Deploys and installs the `msmodeling optix` serving auto-optimization CLI tool, including repository checking, pip installation, and CLI verification.
- `msmodeling-optix-param-recommend`: Recommends MindIE/vLLM serving auto-optimization parameters, search ranges, benchmark configurations, and `config.toml` snippets based on hardware, model, workload, and optimization objective.

## 3. Core Capabilities

- Explain the boundaries and typical usage of the currently integrated LLM simulation modeling and deployment planning capabilities.
- Organize the key input parameters for the `text_generate`, `throughput_optimizer`, device profile, and serving auto-optimization deployment scenarios.
- Help you distinguish the supported task types, including environment initialization, single-point verification, throughput planning, device modeling, serving auto-optimization deployment, and parameter recommendation.
- Provide candidate commands, input lists, check items, and verification paths based on the local repository documentation and implementation.
- Clearly distinguish between simulation or planning suggestions and measured conclusions when no real run results are available.

## 4. Applicable Scenarios

- You want to initialize the msModeling development environment, create `myenv`, or install `requirements.txt` following the README.
- You want to configure `PYTHONPATH` and `HF_ENDPOINT` for the current session, or check Python dependency consistency.
- You want to determine whether a msModeling request should follow the single-point simulation, throughput planning, or device modeling process.
- You want to organize the parameters to prepare before running `python -m cli.inference.text_generate`.
- You want to evaluate the planning approach or input items for `python -m cli.inference.throughput_optimizer`.
- You want to prepare a device profile for new hardware.
- You want to deploy and install the `msmodeling optix` serving auto-optimization tool.
- You want to get MindIE/vLLM serving auto-optimization parameter recommendations and `config.toml` configuration based on hardware, model, workload, and optimization objective.

## 5. Recommended Usage

- Provide the msModeling repository path, model name, target device, number of devices, and task objective directly when possible.
- If you want to initialize the environment, specify whether to use the default `myenv`, whether to allow installing `uv` and `requirements.txt`, and whether to set `PYTHONPATH` or `HF_ENDPOINT`.
- If you need single-point simulation, specify the model, device profile, number of devices, input and output lengths, and prefill/decode mode.
- If you need throughput planning, specify the model, hardware, number of devices, input and output lengths, SLO, and deployment mode (aggregation, separation, or P:D ratio).
- If you want to deploy the serving auto-optimization tool, specify whether to use the default Aliyun PyPI mirror and whether your runtime environment is an Ascend inference product.
- If you want to get serving auto-optimization parameter recommendations, provide as much of the following as possible: inference framework (MindIE/vLLM), hardware information (per-device memory, `world_size`, devices per node, number of nodes), the model `config.json` path or model name, workload token length, and optimization objective (`throughput`/`ttft`/`tpot`/`balanced`).

## 6. Current Boundaries

- For the integrated dedicated skills, the agent can directly handle the following:
  - Environment dependency installation and checking
  - Progressive parameter completion
  - Candidate command generation
  - Pre-execution confirmation
  - Post-execution result summary
  - Deployment, installation, and verification of the serving auto-optimization tool
  - Optimization parameter recommendations based on historical experience and heuristic rules, and `config.toml` configuration
- When environment installation tasks (including serving auto-optimization tool deployment) involve network-based installation, deleting or overwriting an existing environment, or uninstalling Python packages, you must first explain the impact and obtain confirmation.
- For optimization parameter recommendations, the agent first matches configurations from historical hands-on experience and directly adopts the experience values when a match is found. Parts beyond the experience range fall back to general inference from the script. The agent clearly marks the source of both.
- Chains that are not currently integrated fall outside the scope of this agent. The agent must not fabricate automation processes or claim that it can execute them directly.
- If you need real performance results, still rely on the output of actual runs and verification.

## 7. Typical Use Scenarios

| Scenario | Example Prompt |
|---|---|
| Environment initialization | `Please initialize the environment following the msmodeling README, create myenv, and install requirements.txt.` |
| Single-point simulation parameter organization | `Please help me identify the parameters to fill in to run text_generate for Qwen3-32B on Atlas A3 products.` |
| Throughput planning consultation | `I want to compare the deployment throughput planning of the same model on two types of hardware. How should I prepare the throughput_optimizer input?` |
| Device modeling entry | `I want to create a device profile for new hardware. Please first confirm which specifications are required.` |
| Serving auto-optimization deployment | `Please install the msmodeling optix optimization tool and verify that it is available.` |
| Optimization parameter recommendation | `I am deploying Qwen3-32B on Atlas A3 products with vLLM and prioritizing throughput. Please recommend optimization parameters and config.toml configuration.` |
