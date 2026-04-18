# LingBot / LingBot-Stage1 on VideoPhy-2: Standard Evaluation Protocol

## Goal

This document defines the correct evaluation protocol if we want to make a benchmark-style claim such as:

- "LingBot-base is evaluated on VideoPhy-2"
- "LingBot-Stage1 is evaluated on VideoPhy-2"

The key point is that **VideoPhy-2 is a benchmark and evaluation protocol**, not just an auto-rater. A valid VideoPhy-2 result must preserve the benchmark's input assumptions and fairness constraints.

## What counts as a standard VideoPhy-2 evaluation

A run qualifies as a standard VideoPhy-2 evaluation only if all of the following are true:

1. The prompts come from the official VideoPhy-2 benchmark split.
2. Each model generates videos for the same prompt list and the same seed list.
3. The generation setup is matched across models:
   - same resolution
   - same number of frames
   - same sampling steps
   - same guidance settings if comparable
4. No model receives hidden oracle information derived from benchmark ground truth.
5. The generated videos are evaluated with the same judge:
   - human evaluation, or
   - a trustworthy VideoPhy-2 AutoEval configuration
6. The reported metrics are:
   - `SA`
   - `PC`
   - `joint = fraction(SA >= 4 and PC >= 4)`

The official VideoPhy-2 inference interface reflects this setup:

- `SA` uses `videopath + caption`
- `PC` uses `videopath`

See [README.md](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/videophy/VIDEOPHY2/README.md#L59).

## Why this is non-trivial for LingBot

Our model family is not a plain prompt-only text-to-video model.

The current Stage-1 data and helper stack use much more than text:

- `prompt`
- video tensor
- `poses`
- `actions`
- `intrinsics`

See [stage1_components.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/stage1_components.py#L77).

The helper also constructs generation-side conditioning from:

- text context from T5
- first-frame latent derived from the video
- control signal derived from camera poses, actions, and intrinsics

See:

- [stage1_components.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/stage1_components.py#L187)
- [stage1_components.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/stage1_components.py#L197)
- [stage1_components.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/stage1_components.py#L213)

This means we must explicitly separate:

- **standard benchmark evaluation**
- **internal conditioned evaluation**

Otherwise we risk reporting a non-comparable result as if it were a standard VideoPhy-2 number.

## Two valid evaluation tracks

### Track A: Standard VideoPhy-2 benchmark track

This is the only track that should be described as:

- "evaluated on VideoPhy-2"
- "benchmark result on VideoPhy-2"

In this track, every model may only use conditioning that is available from the benchmark input itself or from model-internal computation.

Allowed inputs:

- official VideoPhy-2 prompt
- random seed
- model-internal text embeddings derived from the prompt
- fixed global decoding hyperparameters
- optional fixed non-oracle defaults shared by all prompts and all compared models

Forbidden inputs:

- benchmark ground-truth video
- first frame extracted from benchmark ground-truth video
- ground-truth camera trajectory
- ground-truth actions
- ground-truth intrinsics
- future states or any annotation unavailable at inference time
- any privileged conditioning available to one model but not another

If LingBot cannot generate at all without oracle controls, then LingBot in its current form is **not directly eligible** for Track A.

### Track B: Internal conditioned VideoPhy-style track

This track is still useful scientifically, but it must not be presented as a standard VideoPhy-2 benchmark result.

This is the correct label when we evaluate:

- prompt + additional control signals
- prompt + camera/action conditioning
- prompt + first-frame conditioning
- prompt + any dataset-specific privileged signals

Recommended naming:

- "VideoPhy-style conditioned evaluation"
- "Internal physical-consistency evaluation using VideoPhy-2 prompts"
- "Conditioned variant on VideoPhy-2 prompt set"

Do not call this:

- "VideoPhy-2 benchmark result"
- "standard VideoPhy-2 evaluation"

## Condition taxonomy for LingBot

Use the following decision rule for every model input.

### Allowed in standard benchmark track

- Prompt text
- Seed
- Model architecture internals
- Text encoder context derived only from the prompt
- Fixed, prompt-independent default camera/control prior if shared across all models and samples

### Not allowed in standard benchmark track

- First frame taken from the target video
- Ground-truth clip frames
- `poses.npy`
- `action.npy`
- `intrinsics.npy`
- Any control signal computed from ground-truth trajectory or video

### Conditionally allowed only in internal track

- Retrieved first frame from an external public source
- Retrieved camera trajectory from an external controller
- Heuristic action prior generated by another model

These are not oracle by default, but they still change the task from prompt-only text-to-video into conditioned video generation. They therefore belong to Track B unless the benchmark itself defines them as part of the input.

## Correct standard protocol for LingBot-base and LingBot-Stage1

If we want a real VideoPhy-2 benchmark comparison, the correct protocol is:

### Step 1: Fix the benchmark prompt set

Use the official VideoPhy-2 prompt split as the manifest.

Each row should contain at least:

- `sample_id`
- `prompt`

Optional:

- official metadata fields if provided by the benchmark

Do not use our current custom CSGO manifest as the benchmark prompt source.

Our current wrapper can build the AutoEval CSV once a manifest exists, but it currently assumes a local manifest row field called `prompt` for `SA`:

- [videophy2.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/eval/videophy2.py#L80)

### Step 2: Define the admissible inference input

For both LingBot-base and LingBot-Stage1, define a benchmark-safe inference mode that consumes:

- prompt only
- seed
- benchmark-approved public defaults

and does **not** consume:

- GT first frame
- GT pose
- GT action
- GT intrinsics

If such a mode does not exist yet, it must be implemented before claiming standard VideoPhy-2 results.

### Step 3: Generate videos under matched settings

For each model:

- same prompt list
- same seeds, e.g. `42`, `123`, `3407`
- same frame count
- same resolution
- same sample steps
- same guidance schedule when meaningful

Recommended output structure:

- `runs/eval/<experiment_name>/seed_<seed>/csgo_metrics/videos/<sample_id>_gen.mp4`

This matches our current wrapper expectation for generated outputs:

- [videophy2.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/eval/videophy2.py#L69)
- [run_videophy2_lingbot_parallel.sh](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_videophy2_lingbot_parallel.sh#L116)

### Step 4: Prepare VideoPhy-2 evaluation CSVs

For each generated run and seed:

- `SA` CSV contains:
  - `videopath`
  - `caption`
- `PC` CSV contains:
  - `videopath`

This is already how our wrapper works:

- [build_videophy2_input_csv](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/eval/videophy2.py#L51)

### Step 5: Score with a valid judge

Preferred order:

1. Human evaluation on a representative subset
2. VideoPhy-2 AutoEval, only if it passes sanity checks on the current environment

Because our current AutoEval path has shown collapse behavior on H20, any benchmark-quality claim should currently prioritize human evaluation unless the AutoEval judge is revalidated.

### Step 6: Report metrics

For each model and seed:

- `SA mean`
- `PC mean`
- `joint`

Then aggregate over seeds:

- mean across seeds
- count of valid samples

Our wrapper summary logic already computes these:

- [summarize_videophy2_outputs](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/eval/videophy2.py#L145)

## Minimal fairness checklist

Before reporting a number as "VideoPhy-2":

- Same prompt manifest for all compared models
- Same seed list
- Same decoding hyperparameters
- Same frame count and resolution
- No GT-derived control inputs
- Same evaluator for all methods
- If using AutoEval, report parse coverage / valid coverage

If any item fails, the run should be relabeled as:

- ablation
- internal evaluation
- conditioned evaluation

not as a standard benchmark result.

## Recommended evaluation matrix for our project

### Official benchmark table

Include only rows that obey Track A:

- `LingBot-base (prompt-only benchmark-safe mode)`
- `LingBot-Stage1 (prompt-only benchmark-safe mode)`

Metrics:

- `SA`
- `PC`
- `joint`

### Internal conditioned table

Separate table for rows that use privileged or task-specific conditioning:

- `LingBot-base + controls`
- `LingBot-Stage1 + controls`
- `LingBot-Stage1 + TRD + controls`

These rows are still valuable, but they must be labeled as conditioned/internal.

## How this maps to the current repo

The current repo already has a convenient wrapper for comparing base vs stage1:

- [run_videophy2_lingbot_parallel.sh](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_videophy2_lingbot_parallel.sh#L19)

It currently schedules:

- `exp_base_zeroshot`
- `exp_stage1_epoch2`

across seeds:

- `42`
- `123`
- `3407`

However, the current default manifest is:

- `data/manifests/csgo_phys_val50.csv`

See [run_videophy2_lingbot_parallel.sh](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_videophy2_lingbot_parallel.sh#L20).

So out of the box, the current script is **not yet a standard VideoPhy-2 benchmark runner**. It is a useful wrapper, but to become a true benchmark protocol it needs:

1. official VideoPhy-2 prompt manifest
2. benchmark-safe LingBot inference mode
3. validated judge

## Final decision rule

Use this rule when writing the paper or report:

- If the model uses only prompt plus benchmark-safe public inputs:
  - report as **VideoPhy-2 benchmark**
- If the model uses prompt plus extra task-specific controls:
  - report as **conditioned internal evaluation on VideoPhy-2 prompts**
- If the judge itself is unstable or collapsed:
  - do not use AutoEval as the main result
  - fall back to human evaluation or another validated metric

## Recommended next implementation step

Before running a formal benchmark experiment, implement one concrete artifact:

- a **benchmark-safe LingBot inference mode** that accepts only prompt-level inputs

Without that mode, the correct protocol for our current system is Track B, not Track A.
