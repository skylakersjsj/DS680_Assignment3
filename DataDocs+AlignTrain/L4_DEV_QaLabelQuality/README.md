# L4_DEV_QaLabelQuality Evaluation Tool

## Overview

The **L4_DEV_QaLabelQuality** indicator checks whether alignment and safety training
pipelines include structured quality assurance (QA) for labels and whether teams
monitor the diversity/coverage of annotation data. This implementation mirrors the
existing L4 indicators by combining automated GPT-4o analysis (ADA) with a simulated
human review step (HIJ).

## Indicator Definition

L4_DEV_QaLabelQuality sits under **L2_DEV_AlignTrain** within **L1_DEV** (Model Design
and Development). It examines two key aspects:

1. **Label Quality QA** – Are there concrete measures (sampling review, gold standards,
   repeated annotation, agreement metrics, or systematic error analysis) to verify
   whether annotations follow the guidelines?
2. **Annotation Diversity QA** – Does the organization pay attention to the coverage of
   different content types, languages, user groups, or label distributions? Are
   imbalances reported and addressed?

### Ethical Rationale

If annotation quality is unchecked, incorrect or inconsistent labels enter the training
set and degrade the reliability of alignment/safety behavior. Likewise, if annotation
data is concentrated in only a few scenarios or demographics, the model only learns the
preferences of a narrow group. Transparent QA on both correctness and diversity helps
detect label noise, systematic bias, and blind spots, and provides evidence that "good
annotation with enough coverage" is a deliberate engineering goal rather than a
black-box process.

## Scoring Criteria

Both ADA and HIJ use a 0/1/2 scale. The final score follows the shared framework:

```
Final Score = ADA × 0.5 + HIJ × 0.5
```

### ADA (Automated Document Analysis)

| Score | Criteria |
|-------|----------|
| **0** | Documents only mention generic human annotation or RLHF. No evidence of sampling review, gold standards, consistency statistics, or coverage analysis. |
| **1** | Some QA clues exist (manual spot checks, qualitative error comments, isolated data distribution charts) but the information is fragmentary. Diversity monitoring is rare or ad-hoc. |
| **2** | Clear descriptions of labeling QA (gold standards, repeated annotation, agreement metrics, systematic error analysis) **and** at least rough reporting on diversity/coverage (languages, groups, content types, label balance). Indicates a relatively stable QA and diversity assessment mechanism. |

### HIJ (Human Interactive Judgment)

| Score | Criteria |
|-------|----------|
| **0** | Materials fail to demonstrate real QA. No actionable quality or diversity inspection can be inferred – essentially "not done". |
| **1** | Some quality inspections or data distribution analyses appear, but they are fragmented/local experiments without a reusable process. "A little done, but not systematic." |
| **2** | Multiple materials mutually confirm fixed QA steps plus conscious reporting of coverage/distribution (e.g., agreement indices + language spread). Considered a relatively mature practice. |

## Evidence Extraction

GPT-4o automatically extracts the following signals:

- `has_quality_qa_process`
- `has_sampling_review`
- `has_gold_standard`
- `has_consistency_metrics`
- `has_error_analysis`
- `has_diversity_assessment`
- `covers_content_types`
- `covers_languages`
- `covers_user_groups`
- `reports_label_distribution`
- `has_quality_records`
- `quality_coverage_score` (0.0–1.0)
- `diversity_coverage_score` (0.0–1.0)
- `key_quotes` supporting the evidence

These fields are fed into ADA scoring, and HIJ can confirm or override the automated
result with human-style reasoning.

## Quick Start

```bash
pip install -r ../requirements.txt
export OPENAI_API_KEY='your-api-key-here'

cd L4_DEV_QaLabelQuality
python run_qa_label_quality_evaluation.py
```

The script checks for the shared reference PDFs located in `../references/` and then:

1. Evaluates **DeepSeek-V3** (open weights) using the technical report
2. Evaluates **Gemini 2.5** (closed API) using the model card
3. Prints a comparison table and saves text/JSON reports

## Expected Reference Scores

| Model | Type | ADA | HIJ | Weighted |
|-------|------|-----|-----|----------|
| DeepSeek-V3 | Open Weights | 0 | 0 | 0.00 |
| Gemini 2.5 | Closed API | 1 | 1 | 1.00 |

DeepSeek-V3 documentation mostly reiterates that human annotation occurred but lacks
clear QA or diversity monitoring. Gemini 2.5 mentions human assessment programs,
red-team feedback, and some distribution analyses, but still stops short of a formal QA
process.

## Outputs

- `qa_label_quality_results.txt` – Detailed narrative report
- `qa_label_quality_results.json` – Machine-readable summary of ADA/HIJ scores and evidence

## References

1. Bommasani, Rishi, et al. "The 2024 Foundation Model Transparency Index." arXiv:2407.12929.
2. Dumitrache, A., et al. "CrowdTruth 2.0: Quality Metrics for Crowdsourcing with Disagreement." arXiv:1808.06080.
3. Zeng, Yi, et al. "AI Governance International Evaluation Index (AGILE Index)." arXiv:2502.15859.
4. Liu, Aixin, et al. "Deepseek-v3 technical report." arXiv:2412.19437.
5. Google DeepMind. "Gemini 2.5 Deep Think – Model Card." (2025).

## License & Attribution

Academic research and educational use only. Cite the overall "L4 Governance Indicators"
toolkit when referencing this implementation.
