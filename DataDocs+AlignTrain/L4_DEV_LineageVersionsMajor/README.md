# L4_DEV_LineageVersionsMajor Evaluation Tool

## Overview

This project implements the **L4_DEV_LineageVersionsMajor** indicator, powered by **OpenAI GPT-4o**, for evaluating whether foundation model developers systematically track the lineage and versions of major datasets.

## What is L4_DEV_LineageVersionsMajor?

L4_DEV_LineageVersionsMajor is an indicator in the data governance subdomain of Model Design and Development (L1_DEV). It measures whether developers systematically track the genealogy and versions of major datasets that support chat-based models.

### Key Aspects

1. **Dataset Registration and Management**
   - Are major datasets registered and managed as independent objects?
   - Do they have clear names or identifiers?
   - Are they linked to specific internal version labels?

2. **Lineage Information Recording**
   - Are upstream data sources documented?
   - Are key cleaning and filtering steps recorded?
   - Are major differences between dataset versions documented?
   - Can you map which dataset version corresponds to which model version?

### Major Datasets

This indicator focuses on core datasets including:

- **Pre-training data**: Large-scale corpus for initial model training
- **Instruction/alignment fine-tuning data**: Data for instruction following and alignment
- **Safety and red-team evaluation data**: Data for safety testing and adversarial evaluation

### Ethical Rationale

Without tracking dataset lineage and versions, when upstream data is found to contain bias, infringement, or privacy risks, it becomes difficult to:

- Determine whether the problematic data entered the model
- Identify which model versions were affected
- Perform timely and targeted repairs or model retirement

Establishing clear lineage and version records enables:

- **Accountability tracing**: Match model outputs to specific data sources and processing decisions
- **Risk analysis**: Understand data provenance and potential issues
- **Fairness research**: Investigate bias sources
- **Rights requests**: Respond to deletion requests and compliance requirements

## Scoring Criteria

### ADA (Automated Document Analysis)

| Score | Criteria |
|-------|----------|
| **0** | Only general statements like "using large-scale internet data" or "mixing multiple public and authorized data". No named major datasets, data pipeline, source list, or version information. Cannot determine if lineage or version tracking exists. |
| **1** | Some lineage-related information visible (several core data sources listed, or brief data processing procedures). However, major datasets lack clear names or IDs, cannot see correspondence with specific model versions, no systematic "dataset version table" or similar structure. |
| **2** | Named/numbered major datasets appear with their versions or batches mentioned. At least basic lineage relationships (upstream sources, key cleaning/filtering steps) and major differences between versions provided. Can roughly see which dataset version corresponds to which model versions. |

### HIJ (Human Interactive Judgment)

| Score | Criteria |
|-------|----------|
| **0** | Documentation basically limited to "vague explanation of data sources". Cannot see reliable lineage or version records of major datasets. Cannot indicate there is a real tracking mechanism. |
| **1** | Some lineage information confirmed (specific descriptions of sources and processing of some core corpus), but information is fragmented, coverage is limited, lacks clear version mapping rules. Can only be regarded as "conscious, but not systematic". |
| **2** | Evidence of mutual verification seen in multiple materials: major datasets have fixed names or IDs, version or batch records, key processing steps and correspondence with model versions are relatively clear. Can be considered to have established relatively stable lineage and version tracking practices. |

### Final Score Calculation

```text
Final Score = ADA Score × 0.5 + HIJ Score × 0.5
```

## Key Features

- ✨ **GPT-4o Powered**: Advanced AI for intelligent lineage and version analysis
- 📄 **PDF Support**: Automatic extraction and analysis of PDF documents
- 🔍 **Lineage Detection**: Identifies dataset names, versions, sources, and processing steps
- 🤖 **Automated Scoring**: Hybrid ADA + HIJ evaluation
- 📊 **Comprehensive Reports**: Detailed reports with evidence and reasoning
- 💾 **Multiple Outputs**: Text and JSON format results

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Running Evaluation

```bash
python run_lineage_evaluation.py
```

## Evaluation Evidence

GPT-4o analyzes documents to extract evidence for:

| Evidence Type | Description |
|--------------|-------------|
| **has_dataset_names** | Are major datasets given clear names? |
| **has_dataset_ids** | Are datasets identified with specific IDs? |
| **has_version_labels** | Are there version labels for datasets? |
| **has_upstream_sources** | Is upstream source information provided? |
| **has_processing_steps** | Are key data processing steps described? |
| **has_version_differences** | Are differences between versions documented? |
| **has_model_dataset_mapping** | Can you map dataset versions to model versions? |
| **has_data_pipeline** | Is there a description of data pipeline/lineage flow? |
| **named_datasets_count** | How many major datasets are explicitly named? |
| **versioned_datasets_count** | How many named datasets have version information? |

## Usage Example

```python
from L4_DEV_LineageVersionsMajor_GPT4o import L4_DEV_LineageVersionsMajor_GPT4o

# Initialize evaluator
evaluator = L4_DEV_LineageVersionsMajor_GPT4o()

# Evaluate a model
result = evaluator.evaluate_model(
    model_name="MyModel",
    model_type="Open Weights",
    document_sources=["path/to/technical_report.pdf"],
    use_hij=True,
    additional_context="Additional context for evaluation"
)

# View results
print(result)
print(f"Final Score: {result.weighted_score:.2f}")

# Generate comparison table
print(evaluator.generate_comparison_table())
```

## Example Results

Based on the article's analysis, expected scores for evaluated models:

```text
===================================================================================
Model                Type            ADA      HIJ      Weighted
-----------------------------------------------------------------------------------
DeepSeek-V3          Open Weights    0        1        0.50
Gemini 2.5           Closed API      1        1        1.00
===================================================================================
```

### Key Findings

- **DeepSeek-V3** (0.50): Limited lineage tracking. Automatic analysis finds hardly any formed "dataset genealogy" or version marking. Manual review can piece together scattered source and processing clues, but difficult to answer which version of core datasets is bound to a specific model version.

- **Gemini 2.5** (1.00): More concentrated and structured. Description of core data sources, cleaning processes, and data update rhythm is relatively more systematic. Automatic analysis can identify certain degree of lineage information, and manual review can find mutual verification across materials. Shows awareness that major datasets should be managed as "versioned and historical" objects.

## Project Structure

```text
p4/
├── L4_DEV_LineageVersionsMajor_GPT4o.py    # Main evaluation tool
├── run_lineage_evaluation.py                # Automated evaluation script
├── README_LineageVersionsMajor.md           # This documentation
├── requirements.txt                         # Dependencies (shared)
└── references/                              # Reference documents (shared)
    ├── deepseek_v3.pdf
    ├── gemini_2_5_model_card.pdf
    └── fmti_2024.pdf
```

## Comparison with L4_DEV_DatasheetsData

| Aspect | L4_DEV_DatasheetsData | L4_DEV_LineageVersionsMajor |
|--------|----------------------|----------------------------|
| **Focus** | Documentation existence and versioning | Lineage tracking and version mapping |
| **Key Question** | Are there datasheets for datasets? | Can you trace data origins and versions? |
| **Evaluation** | Checks for datasheets, version tags | Checks for lineage flow, processing steps |
| **Goal** | Ensure datasets are documented | Ensure datasets are traceable |

Both indicators complement each other in assessing comprehensive data governance practices.

## Theoretical Foundation

1. **Bommasani, R., et al.** (2024). "The 2024 Foundation Model Transparency Index." arXiv:2407.12929
   - Framework for evaluating transparency practices

2. **González-Cebrián, A., et al.** (2024). "Standardised versioning of datasets: a FAIR-compliant proposal." Scientific Data 11.1: 358
   - Standardized dataset versioning framework

3. **Zeng, Y., et al.** (2025). "AI Governance International Evaluation Index (AGILE Index)." arXiv:2502.15859
   - AI governance evaluation methodology

4. **Liu, A., et al.** (2024). "Deepseek-v3 technical report." arXiv:2412.19437
   - DeepSeek-V3 technical details

5. **Google DeepMind.** (2025). "Gemini 2.5 Deep Think – Model Card."
   - Gemini 2.5 official documentation

## Output Formats

### Comparison Table

```text
===================================================================================
Table: L4_DEV_LineageVersionsMajor Comparative Assessment (GPT-4o Powered)
===================================================================================
Model                Type            ADA      HIJ      Weighted
-----------------------------------------------------------------------------------
DeepSeek-V3          Open Weights    0        1        0.50
Gemini 2.5           Closed API      1        1        1.00
===================================================================================
```

### Detailed Results

Each evaluation includes:

- ADA and HIJ scores
- Weighted final score
- Lineage evidence extracted by GPT-4o
- GPT-4o reasoning process
- HIJ review rationale
- Key quotes from documents

### JSON Output

```json
{
  "indicator": "L4_DEV_LineageVersionsMajor",
  "description": "Dataset lineage and version tracking evaluation",
  "models": [
    {
      "name": "DeepSeek-V3",
      "type": "Open Weights",
      "ada_score": 0,
      "hij_score": 1,
      "weighted_score": 0.5,
      "evidence": {...},
      "hij_rationale": "...",
      "gpt4o_reasoning": "..."
    }
  ]
}
```

## Advanced Usage

### Batch Evaluation

```python
evaluator = L4_DEV_LineageVersionsMajor_GPT4o()

models = [
    ("Model-A", "Open Weights", ["docs/model_a.pdf"]),
    ("Model-B", "Closed API", ["docs/model_b.pdf"]),
]

for name, model_type, docs in models:
    evaluator.evaluate_model(name, model_type, docs)

print(evaluator.generate_comparison_table())
```

### Skip HIJ for Faster Evaluation

```python
result = evaluator.evaluate_model(
    model_name="MyModel",
    model_type="Open Weights",
    document_sources=["report.pdf"],
    use_hij=False  # Skip HIJ to save API calls
)
```

## Cost Estimation

Using GPT-4o for evaluation:

- **Small documents** (10-20 pages): ~$0.05-0.10 per model
- **Medium documents** (50-100 pages): ~$0.20-0.40 per model
- **Large documents** (200+ pages): ~$0.50-1.00 per model

Each evaluation includes 3 API calls:

1. Lineage evidence extraction
2. ADA scoring
3. HIJ scoring (optional)

## Troubleshooting

### Common Issues

**Issue**: API key error

```bash
export OPENAI_API_KEY='your-actual-api-key'
```

**Issue**: PDF extraction failed

- Ensure PDF is not scanned (requires OCR)
- Try converting to text file
- Check if PDF file is corrupted

**Issue**: Different scores than expected

GPT-4o may interpret evidence differently based on:

- How lineage information is presented in documents
- Clarity of dataset naming and versioning
- Completeness of processing step descriptions

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{l4_dev_lineage_versions_major_2025,
  title={L4\_DEV\_LineageVersionsMajor: Dataset Lineage and Version Tracking Evaluation Tool},
  author={Zhang, Qurui},
  year={2025},
  note={GPT-4o Enhanced Implementation}
}
```

## Related Indicators

This indicator belongs to the AI governance evaluation framework:

- **L1_DEV**: Model Design and Development
  - **L4**: Data Documentation subdomain
    - **L4_DEV_DatasheetsData**: Datasheets and documentation
    - **L4_DEV_LineageVersionsMajor**: Lineage and version tracking (this indicator)

## License

This project is for academic research and educational purposes.

---

**Last Updated**: 2025-12-10
**Version**: 1.0.0 (GPT-4o Enhanced)
