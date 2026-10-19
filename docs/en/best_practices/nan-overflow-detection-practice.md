# Precision Issues: Best Practices for NaN Overflow

This section uses a model that shows NaN overflow in the first training step on 512 devices as an example to demonstrate how to complete source device localization and root-cause operator analysis for NaN overflow end-to-end with the precision localization feature module of msagent.

## 1. Background

A model shows NaN overflow in the first training step on 512 devices. You need to locate the source device and the root-cause operator.

## 2. Localization Steps

### 2.1 Process

1. Prepare the dump data to be analyzed.
2. Access the precision localization feature module of msagent and enter the initial prompt: "Based on the input training dump data, analyze the NaN overflow in it and identify the source device and the root-cause operator. The data path is xxx. Before you start the analysis, discuss the analysis approach and plan with me first."
3. Discuss and determine the analysis steps: locate the source device first, and then locate the root-cause operator.
4. Obtain the analysis conclusion of the agent: the source devices are rank 168 and rank 325, and the root-cause operator is the `npu_grouped_matmul` operator.
5. Verify the analysis conclusion by confirming in the data that the input of `NPU.npu_grouped_matmul.61.forward` is not NaN, the output is NaN, no NaN occurs before this operator, and NaN occurs after this operator.
6. Generate the analysis report.

### 2.2 Demonstration

#### 2.2.1 Describing the Goal and the Data

![Describe the goal and the data](../figures/best_practices/nan_prompt.jpg)

#### 2.2.2 Discussing and Determining the Analysis Steps

![Discuss and determine the analysis steps](../figures/best_practices/nan_plan.jpg)

#### 2.2.3 Obtaining the Analysis Conclusion of the Agent

![Obtain the analysis conclusion of the agent](../figures/best_practices/nan_result.jpg)

#### 2.2.4 Verifying That the Input Is Not NaN

![The input is not NaN](../figures/best_practices/nan_dump_1.jpg)

#### 2.2.5 Verifying That the Output Is NaN

![The output is NaN](../figures/best_practices/nan_dump_2.jpg)

#### 2.2.6 Obtaining the Analysis Report

![Analysis report](../figures/best_practices/nan_report.jpg)
