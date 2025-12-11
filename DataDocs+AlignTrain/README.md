# L4 Data Governance Indicators

Five GPT-4o powered indicators for evaluating data governance, alignment, and responsible-use practices in foundation model development.

## Indicators

### 1. L4_DEV_DatasheetsData
Evaluates whether developers have established internal datasheets/data statements for key datasets with version-based management.

**Location**: `L4_DEV_DatasheetsData/`

**Expected Scores**:
- DeepSeek-V3: ADA=1, HIJ=1, Weighted=1.00
- Gemini 2.5: ADA=1, HIJ=2, Weighted=1.50

### 2. L4_DEV_LineageVersionsMajor
Evaluates whether developers systematically track the lineage and versions of major datasets.

**Location**: `L4_DEV_LineageVersionsMajor/`

**Expected Scores**:
- DeepSeek-V3: ADA=0, HIJ=1, Weighted=0.50
- Gemini 2.5: ADA=1, HIJ=1, Weighted=1.00

### 3. L4_DEV_AnnotatorGuidelines
Evaluates whether developers have formed comprehensive and reusable annotation guidelines for security-related annotation work.

**Location**: `L4_DEV_AnnotatorGuidelines/`

**Expected Scores**:
- DeepSeek-V3: ADA=0, HIJ=1, Weighted=0.50
- Gemini 2.5: ADA=1, HIJ=2, Weighted=1.50

### 4. L4_DEV_QaLabelQuality
Evaluates whether alignment/safety training includes structured QA for labeling quality and diversity coverage checks.

**Location**: `L4_DEV_QaLabelQuality/`

**Expected Scores**:
- DeepSeek-V3: ADA=0, HIJ=0, Weighted=0.00
- Gemini 2.5: ADA=1, HIJ=1, Weighted=1.00

### 5. L4_DEV_EduGoalSpec
Evaluates whether safe-use education/training has explicit goals, KPIs, baselines, and assessment plans.

**Location**: `L4_DEV_EduGoalSpec/`

**Expected Scores**:
- DeepSeek-V3: ADA=0, HIJ=0, Weighted=0.00
- Gemini 2.5: ADA=0, HIJ=1, Weighted=0.50

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
export OPENAI_API_KEY='your-api-key-here'
```

### Run Evaluations

**Indicator 1: DatasheetsData**
```bash
cd L4_DEV_DatasheetsData
python run_evaluation.py
```

**Indicator 2: LineageVersionsMajor**
```bash
cd L4_DEV_LineageVersionsMajor
python run_lineage_evaluation.py
```

**Indicator 3: AnnotatorGuidelines**
```bash
cd L4_DEV_AnnotatorGuidelines
python run_annotator_evaluation.py
```

**Indicator 4: QaLabelQuality**
```bash
cd L4_DEV_QaLabelQuality
python run_qa_label_quality_evaluation.py
```

**Indicator 5: EduGoalSpec**
```bash
cd L4_DEV_EduGoalSpec
python run_edu_goal_spec_evaluation.py
```

## Project Structure

```
p4/
├── L4_DEV_DatasheetsData/          # Indicator 1: Dataset documentation
│   ├── L4_DEV_DatasheetsData_GPT4o.py
│   ├── run_evaluation.py
│   └── README.md
│
├── L4_DEV_LineageVersionsMajor/    # Indicator 2: Dataset lineage
│   ├── L4_DEV_LineageVersionsMajor_GPT4o.py
│   ├── run_lineage_evaluation.py
│   └── README.md
│
├── L4_DEV_AnnotatorGuidelines/     # Indicator 3: Annotation guidelines
│   ├── L4_DEV_AnnotatorGuidelines_GPT4o.py
│   ├── run_annotator_evaluation.py
│   └── README.md
│
├── L4_DEV_QaLabelQuality/          # Indicator 4: Label QA & diversity
│   ├── L4_DEV_QaLabelQuality_GPT4o.py
│   ├── run_qa_label_quality_evaluation.py
│   └── README.md
│
├── L4_DEV_EduGoalSpec/             # Indicator 5: Education goal specification
│   ├── L4_DEV_EduGoalSpec_GPT4o.py
│   ├── run_edu_goal_spec_evaluation.py
│   └── README.md
│
├── references/                      # Shared reference PDFs
│   ├── deepseek_v3.pdf
│   ├── gemini_2_5_model_card.pdf
│   ├── fmti_2024.pdf
│   ├── gonzalez_cebrian_2024_dataset_versioning.pdf
│   └── safe_rlhf_2023.pdf
│
└── requirements.txt                 # Shared dependencies
```

## Reference Documents

All indicators share the reference PDFs in `references/`:

- **deepseek_v3.pdf** (1.8 MB) - DeepSeek-V3 technical report
- **gemini_2_5_model_card.pdf** (1.4 MB) - Gemini 2.5 model card
- **fmti_2024.pdf** (6.1 MB) - Foundation Model Transparency Index 2024
- **gonzalez_cebrian_2024_dataset_versioning.pdf** (97 KB) - Dataset versioning standards
- **safe_rlhf_2023.pdf** (1.0 MB) - Safe RLHF framework

## Indicator Comparison

| Indicator | Domain | Focus | DeepSeek-V3 | Gemini 2.5 |
|-----------|--------|-------|-------------|------------|
| **DatasheetsData** | Data Documentation | Dataset documentation existence | 1.00 | 1.50 |
| **LineageVersionsMajor** | Data Governance | Dataset lineage tracking | 0.50 | 1.00 |
| **AnnotatorGuidelines** | Alignment & Training | Annotation guidelines quality | 0.50 | 1.50 |
| **QaLabelQuality** | Alignment & Training | Label QA & diversity monitoring | 0.00 | 1.00 |
| **EduGoalSpec** | Architecture & Objectives | Safe-use education goals/KPIs | 0.00 | 0.50 |

## Evaluation Framework

All indicators use the same hybrid methodology:

```
Final Score = ADA Score × 0.5 + HIJ Score × 0.5
```

Where:
- **ADA** (Automated Document Analysis): GPT-4o analyzes documents automatically
- **HIJ** (Human Interactive Judgment): GPT-4o simulates expert human review

### Scoring Scale (0/1/2)

- **0**: No evidence or very general statements
- **1**: Some evidence, but fragmented or incomplete
- **2**: Clear, comprehensive, and systematic evidence

## Documentation

Each indicator folder contains:
- **README.md** - Complete documentation for that indicator
- **[Indicator]_GPT4o.py** - Main implementation
- **run_[indicator]_evaluation.py** - Automated evaluation script

## Dependencies

```
openai>=1.0.0
PyPDF2>=3.0.0
```

## Usage Example

```python
# Example: Evaluate all three indicators for a custom model
from L4_DEV_DatasheetsData_GPT4o import L4_DEV_DatasheetsData_GPT4o
from L4_DEV_LineageVersionsMajor_GPT4o import L4_DEV_LineageVersionsMajor_GPT4o
from L4_DEV_AnnotatorGuidelines_GPT4o import L4_DEV_AnnotatorGuidelines_GPT4o

docs = ["my_model_report.pdf"]

# Evaluate each indicator
datasheet_eval = L4_DEV_DatasheetsData_GPT4o()
lineage_eval = L4_DEV_LineageVersionsMajor_GPT4o()
annotator_eval = L4_DEV_AnnotatorGuidelines_GPT4o()

result1 = datasheet_eval.evaluate_model("MyModel", "Open Weights", docs)
result2 = lineage_eval.evaluate_model("MyModel", "Open Weights", docs)
result3 = annotator_eval.evaluate_model("MyModel", "Open Weights", docs)

print(f"Datasheets: {result1.weighted_score:.2f}")
print(f"Lineage: {result2.weighted_score:.2f}")
print(f"Annotator: {result3.weighted_score:.2f}")
print(f"Average: {(result1.weighted_score + result2.weighted_score + result3.weighted_score) / 3:.2f}")
```

## Cost Estimation

Using GPT-4o for evaluation:
- **Small documents** (10-20 pages): ~$0.05-0.10 per model per indicator
- **Medium documents** (50-100 pages): ~$0.20-0.40 per model per indicator
- **Large documents** (200+ pages): ~$0.50-1.00 per model per indicator

Each evaluation: 3 API calls (evidence extraction, ADA scoring, HIJ scoring)

## Citation

```bibtex
@software{l4_governance_indicators_2025,
  title={L4 Governance Indicators: GPT-4o Enhanced Evaluation Tools},
  author={Zhang, Qurui},
  year={2025},
  note={Five indicators for data governance, alignment, and responsible-use assessment}
}
```

## Theoretical Foundation

1. **Bommasani, R., et al.** (2024). "The 2024 Foundation Model Transparency Index." arXiv:2407.12929
2. **Gebru, T., et al.** (2021). "Datasheets for datasets." Communications of the ACM 64.12: 86-92
3. **González-Cebrián, A., et al.** (2024). "Standardised versioning of datasets." Scientific Data 11.1: 358
4. **Dai, J., et al.** (2023). "Safe RLHF." arXiv:2310.12773
5. **Zeng, Y., et al.** (2025). "AGILE Index." arXiv:2502.15859

## License

For academic research and educational purposes.

---

**Version**: 1.0.0
**Last Updated**: 2025-12-10
**Indicators**: 5 (DatasheetsData, LineageVersionsMajor, AnnotatorGuidelines, QaLabelQuality, EduGoalSpec)
