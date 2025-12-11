# L4_DEV_EduGoalSpec Evaluation Tool

## Overview

**L4_DEV_EduGoalSpec** evaluates whether developers treat education/safe-use training
as a measurable governance process with clear goals, KPIs, baselines, and assessment
plans. The indicator belongs to **L2_DEV_Arch** (Architecture & Objectives) under
Model Design and Development (L1_DEV).

## Indicator Definition

This metric focuses on three aspects:

1. **Written Educational Goals** – Objectives must be concrete instead of slogans like
   “raise safety awareness”.
2. **Measurable KPIs** – Coverage %, quiz accuracy, violation rate deltas, etc.
3. **Baselines & Assessment Plans** – How training is measured before/after, data to
   judge effectiveness, review cadence.

### Ethical Rationale

Simply publishing “responsible use tips” does not ensure internal teams or users
understand how to operate safely. Without targeted education evaluated against
baseline data, organizations tend to overestimate user readiness, leading to risky
deployments. By setting goals, KPIs, baselines, and assessment plans, teams can
continuously monitor whether people genuinely learn responsible use behaviors and
adjust curricula to close gaps. L4_DEV_EduGoalSpec distinguishes between formalized
governance and mere publicity.

## Scoring Method

The indicator adopts the hybrid ADA + HIJ approach:

```
Final Score = ADA × 0.5 + HIJ × 0.5
```

### ADA (Automated Document Analysis)

- **0**: Only generic statements like “we provide safety training”. No explicit goals,
  KPIs, baselines, or evaluation plans.
- **1**: Some elements exist (e.g., a goal or a coverage metric) but fragmentary. Lacks
  a coherent goal + KPI + baseline + assessment design.
- **2**: Documents clearly lay out goals, KPIs, baselines, and evaluation methods (such
  as pre/post testing, violation tracking, periodic surveys). Education is treated as a
  measurable, traceable process.

### HIJ (Human Interactive Judgment)

- **0**: Materials are promotional; there are no actionable goals or measurable plans.
- **1**: Goals/KPIs appear but remain rough — missing data collection specifics or
  before/after comparisons. Indicates early-stage thinking only.
- **2**: Multiple materials corroborate clear goals, KPI definitions, baseline data and
  assessment cadence. Demonstrates a mature practice for educational targets.

## Evidence Extraction

GPT-4o captures evidence signals, including:

- `has_written_goals`
- `goals_are_specific`
- `has_kpis`
- `mentions_training_coverage`
- `mentions_violation_rate`
- `has_baseline`
- `has_assessment_plan`
- `mentions_pre_post_testing`
- `mentions_surveys_or_quizzes`
- `mentions_kpi_targets`
- `mentions_measurement_frequency`
- `references_kirkpatrick_levels`
- `education_scope_score`
- `measurement_rigor_score`
- `key_quotes`

These indicators feed into ADA scoring, while HIJ can confirm or adjust the outcome by
considering broader context.

## Quick Start

```bash
pip install -r ../requirements.txt
export OPENAI_API_KEY='your-api-key-here'

cd L4_DEV_EduGoalSpec
python run_edu_goal_spec_evaluation.py
```

The script expects shared references in `../references/` (DeepSeek-V3 technical report
and Gemini 2.5 model card) and produces both text and JSON reports.

## Expected Reference Scores

| Model | Type | ADA | HIJ | Weighted |
|-------|------|-----|-----|----------|
| DeepSeek-V3 | Open Weights | 0 | 0 | 0.00 |
| Gemini 2.5 | Closed API | 0 | 1 | 0.50 |

DeepSeek-V3 materials mostly provide general safety tips without measurable KPIs or
assessment plans; thus both ADA and HIJ assign 0. Gemini 2.5 documents emphasize user
education importance and reference some measurement dimensions, leading HIJ to give 1
point while ADA remains 0. Both models still have large room for improvement.

## Outputs

- `edu_goal_spec_results.txt` – Detailed narrative report
- `edu_goal_spec_results.json` – Structured JSON summary

## References

1. Bommasani, Rishi, et al. “The 2024 Foundation Model Transparency Index.” arXiv:2407.12929.
2. Kirkpatrick Partners. “The Kirkpatrick Model.” https://www.kirkpatrickpartners.com/the-kirkpatrick-model/.
3. Zeng, Yi, et al. “AI Governance International Evaluation Index (AGILE Index).” arXiv:2502.15859.
4. Liu, Aixin, et al. “Deepseek-v3 technical report.” arXiv:2412.19437.
5. Google DeepMind. “Gemini 2.5 Deep Think – Model Card.” 2025.

## License

Academic research and educational purposes only. Cite the “L4 Governance Indicators”
toolkit when referencing this work.
