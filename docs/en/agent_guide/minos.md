# Minos Document Assistant

`Minos` is an agent designed for document experience and code review assistance. It reviews README files, installation documentation, Quick Start guides, and onboarding paths from the perspective of a newcomer who is interacting with the project for the first time. It also reviews GitCode PRs for change risk and modification quality in the context of the repository.

## Agent Positioning

- Handles README walkthroughs, installation process verification, newcomer onboarding experience reviews, and GitCode PR reviews.
- Focuses on documentation usability, information gaps, implicit prerequisites, and blocker identification, as well as PR regression risk and change completeness.
- Produces document improvement suggestions, code review conclusions, and prioritized checklists for maintainers.

## Core Capabilities

- Actually executes the steps in the README and installation documentation and records blockers.
- Reviews whether the Quick Start is clear enough, reproducible, and newcomer-friendly.
- Distinguishes documentation issues, missing environment prerequisites, and problems with running the project itself.
- Reviews GitCode PRs using PR metadata, diffs, call chains, baseline implementations, and test context.
- Outputs evidence-based document improvement recommendations and code review opinions with severity levels and suggested fixes.

## Recommended Usage

- For document experience tasks, you can directly provide the repository path, the README, or the path of the target document.
- Clearly state the scope you want reviewed, for example installation, startup, verification, troubleshooting, end-to-end onboarding, or GitCode PR risk review.
- If you have a specific user group or review goal, you can also mention it, for example "for developers who are new to Ascend" or "focus on compatibility and test coverage".

## Typical Use Scenarios

| Scenario | Example Prompt | Output Description |
|---|---|---|
| README walkthrough | `Walk through the README completely like a user who is encountering the project for the first time, and point out where you would get stuck.` | Outputs documentation usability conclusions, blockers, evidence, and improvement suggestions. |
| Installation process verification | `Execute the installation steps of this repository by following the installation documentation, and see whether a newcomer can complete them smoothly.` | Outputs missing prerequisites, environment assumptions, and suggested text fixes in the installation process. |
| Quick Start review | `Check whether the Quick Start is enough for a newcomer to get up and running in the shortest possible time, and point out what is still unclear.` | Outputs the getting-started path, comprehension cost, and recommended optimization order. |
| Onboarding experience evaluation | `Evaluate the documentation experience of this repository from an onboarding perspective, and provide a list of improvements that can be put into practice.` | Outputs document improvement suggestions and priorities for maintainers. |
| GitCode PR review | `Review this GitCode PR, focusing on behavioral regression, compatibility, and test coverage.` | Outputs review conclusions with severity, evidence, reasons, and suggested fixes. |
| GitCode PR comment organization | `Check this GitCode PR link and organize the issues that need to be fixed into publishable review comments.` | Outputs item-by-item comment suggestions that can be published directly to the PR. |

For more details, see [`document-ux-review`](../user_guide/document-ux-review.md).
