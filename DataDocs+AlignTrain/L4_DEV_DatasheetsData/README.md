# L4_DEV_DatasheetsData Evaluation Tool

## Overview

This project implements the **L4_DEV_DatasheetsData** indicator, powered by **OpenAI GPT-4o**, for evaluating whether foundation model developers have established internal datasheets/data statements for key datasets with version-based management.

## What is L4_DEV_DatasheetsData?

L4_DEV_DatasheetsData is an indicator in the data documentation subdomain of Model Design and Development (L1_DEV). It measures two critical aspects:

1. **Dataset Documentation**: Whether developers have established internal datasheets/data statements for key datasets (pre-training, instruction/alignment fine-tuning, safety and red-team evaluation datasets)
2. **Version Management**: Whether these documents are updated with major dataset changes and clearly distinguished by version number, date, or effective interval

### Ethical Rationale

Foundation models frequently adjust training and evaluation data during iteration, affecting model outputs. Without corresponding data version records, it becomes difficult to trace specific data updates and responsible parties when bias or security issues arise. Establishing versioned internal datasheets/data statements for key datasets provides:

- Clear records of important data decisions
- Basis for accountability and risk analysis
- Foundation for compliance review
- Transparency for stakeholders

## Key Features

- ✨ **GPT-4o Powered**: Leverages state-of-the-art AI for intelligent document analysis
- 📄 **PDF Support**: Automatic extraction and analysis of PDF documents
- 🤖 **Automated Scoring**: Hybrid evaluation using ADA (Automated Document Analysis) + HIJ (Human Interactive Judgment)
- 📊 **Comprehensive Reports**: Generates detailed evaluation reports and comparison tables
- 💾 **Multiple Output Formats**: Supports both text and JSON output

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Get your API key from: https://platform.openai.com/api-keys

### Running Evaluation

```bash
python run_evaluation.py
```

This will:
1. Analyze DeepSeek-V3 technical report
2. Analyze Gemini 2.5 model card
3. Generate comparative assessment
4. Save results to files

## Evaluation Methodology

### Hybrid Assessment Framework

The evaluation uses a mixed-method approach combining automated analysis and human judgment:

```
Final Score = ADA Score × 0.5 + HIJ Score × 0.5
```

### ADA (Automated Document Analysis)

GPT-4o analyzes documents to extract evidence for:

| Evidence Type | Description |
|--------------|-------------|
| **has_datasheet** | Presence of datasheets/data statements |
| **has_composition_info** | Information about data composition |
| **has_source_info** | Information about data sources |
| **has_intended_use** | Description of intended use |
| **has_limitations** | Known limitations and biases |
| **has_version_info** | Version numbers or identifiers |
| **has_timestamp** | Dates or timestamps |
| **has_change_records** | Changelogs or change records |
| **coverage_ratio** | Proportion of key datasets covered (0.0-1.0) |

**Scoring Criteria:**

- **ADA = 0**: No evidence of datasheets/data statements; only general descriptions
- **ADA = 1**: Some documentation exists with composition/source information, but static with no clear versioning; incomplete coverage
- **ADA = 2**: Clear datasheets/data statements for most/all key datasets; documents have version numbers or time tags; important changes reflected in versions

### HIJ (Human Interactive Judgment)

GPT-4o simulates expert human review, providing a second assessment based on ADA results:

- **HIJ = 0**: No credible evidence of internal datasheets/data statements, let alone version management
- **HIJ = 1**: Some description of key datasets visible, but incomplete coverage; no clear version numbers, timestamps, or change records
- **HIJ = 2**: Key datasets have clear datasheets/data statements with version or time tags; important data changes recorded; stable version management practice evident

## Usage Examples

### Basic Usage

```python
from L4_DEV_DatasheetsData_GPT4o import L4_DEV_DatasheetsData_GPT4o

# Initialize evaluator
evaluator = L4_DEV_DatasheetsData_GPT4o()

# Evaluate a model
result = evaluator.evaluate_model(
    model_name="MyModel",
    model_type="Open Weights",
    document_sources=["path/to/technical_report.pdf"],
    use_hij=True,
    additional_context="Any additional context for evaluation"
)

# View results
print(result)
print(f"Final Score: {result.weighted_score:.2f}")
```

### Batch Evaluation

```python
evaluator = L4_DEV_DatasheetsData_GPT4o()

# Evaluate multiple models
models = [
    ("Model-A", "Closed API", ["model_a_card.pdf"]),
    ("Model-B", "Open Weights", ["model_b_report.pdf"]),
]

for name, model_type, docs in models:
    evaluator.evaluate_model(name, model_type, docs)

# Generate comparison
print(evaluator.generate_comparison_table())
```

## Project Structure

```text
p4/
├── Core Implementation
│   └── L4_DEV_DatasheetsData_GPT4o.py    # Main evaluation tool
│
├── Execution Scripts
│   └── run_evaluation.py                  # Automated evaluation script
│
├── Documentation
│   ├── README.md                          # This file
│   ├── QUICKSTART.md                      # Quick start guide
│   └── PROJECT_SUMMARY.md                 # Project overview
│
├── Configuration
│   ├── requirements.txt                   # Dependencies
│   └── .env.example                       # Environment template
│
└── Reference Documents
    └── references/
        ├── deepseek_v3.pdf               # DeepSeek-V3 technical report
        ├── gemini_2_5_model_card.pdf     # Gemini 2.5 model card
        └── fmti_2024.pdf                 # FMTI 2024 report
```

## Output Formats

### Comparison Table

```text
=====================================================================================
Table: L4_DEV_DatasheetsData Comparative Assessment (GPT-4o Powered)
=====================================================================================
Model                Type            ADA      HIJ      Weighted
-------------------------------------------------------------------------------------
DeepSeek-V3          Open Weights    1        1        1.00
Gemini 2.5           Closed API      1        1        1.00
=====================================================================================
```

### Detailed Results

Each model evaluation includes:
- ADA and HIJ scores
- Weighted final score
- Evidence extracted by GPT-4o
- GPT-4o reasoning process
- HIJ review rationale

### JSON Output

```json
{
  "models": [
    {
      "name": "DeepSeek-V3",
      "type": "Open Weights",
      "ada_score": 1,
      "hij_score": 1,
      "weighted_score": 1.0,
      "evidence": {...},
      "hij_rationale": "...",
      "gpt4o_reasoning": "..."
    }
  ]
}
```

## Cost Estimation

Using GPT-4o for evaluation:
- **Small documents** (10-20 pages): ~$0.05-0.10 per model
- **Medium documents** (50-100 pages): ~$0.20-0.40 per model
- **Large documents** (200+ pages): ~$0.50-1.00 per model

Each evaluation includes 3 API calls:
1. Evidence extraction
2. ADA scoring
3. HIJ scoring (optional)

## Advanced Usage

### ADA Only (Skip HIJ)

```python
result = evaluator.evaluate_model(
    model_name="MyModel",
    model_type="Open Weights",
    document_sources=["report.pdf"],
    use_hij=False  # Skip HIJ to save API calls
)
```

### Limit PDF Pages

```python
# Process only first 50 pages
text = evaluator.load_document("large_report.pdf", max_pages=50)

result = evaluator.evaluate_model(
    model_name="MyModel",
    model_type="Open Weights",
    document_sources=[text]
)
```

### Custom API Key

```python
evaluator = L4_DEV_DatasheetsData_GPT4o(api_key="your-key-here")
```

## Theoretical Foundation

This project is based on the following academic work:

1. **Bommasani, R., et al.** (2024). "The 2024 Foundation Model Transparency Index." arXiv:2407.12929
   - Framework for evaluating transparency practices

2. **Gebru, T., et al.** (2021). "Datasheets for datasets." Communications of the ACM 64.12: 86-92
   - Theoretical foundation for dataset documentation

3. **Zeng, Y., et al.** (2025). "AI Governance International Evaluation Index (AGILE Index)." arXiv:2502.15859
   - AI governance evaluation methodology

4. **Liu, A., et al.** (2024). "Deepseek-v3 technical report." arXiv:2412.19437
   - DeepSeek-V3 technical details

5. **Google DeepMind.** (2025). "Gemini 2.5 Deep Think – Model Card."
   - Gemini 2.5 official documentation

## Troubleshooting

### API Key Error

```text
Error: OpenAI API key not found
```

**Solution:**

```bash
export OPENAI_API_KEY='your-actual-api-key'
```

### PDF Extraction Failed

```text
Error extracting PDF: ...
```

**Solutions:**
- Ensure PDF is not scanned (requires OCR)
- Try converting to text file
- Check if PDF file is corrupted

### GPT-4o Response Error

```text
Error calling GPT-4o API: ...
```

**Solutions:**
- Check API quota
- Verify API key validity
- Check network connection

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{l4_dev_datasheets_data_2025,
  title={L4\_DEV\_DatasheetsData: An Automated Evaluation Tool for Dataset Documentation Practices},
  author={Zhang, Qurui},
  year={2025},
  note={GPT-4o Enhanced Implementation}
}
```

## License

This project is for academic research and educational purposes.

## References

1. Bommasani, Rishi, et al. "The 2024 Foundation Model Transparency Index." arXiv preprint arXiv:2407.12929 (2024).
2. Gebru, Timnit, et al. "Datasheets for datasets." Communications of the ACM 64.12 (2021): 86-92.
3. Zeng, Yi, et al. "AI Governance International Evaluation Index (AGILE Index)." arXiv preprint arXiv:2502.15859 (2025).
4. Liu, Aixin, et al. "Deepseek-v3 technical report." arXiv preprint arXiv:2412.19437 (2024).
5. Google DeepMind. Gemini 2.5 Deep Think – Model Card. 1 Aug. 2025.

---

**Last Updated**: 2025-12-10
**Version**: 1.0.0 (GPT-4o Enhanced)
