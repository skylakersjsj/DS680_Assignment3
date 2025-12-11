# L4_DEV_AnnotatorGuidelines Evaluation Tool

## Overview

This project implements the **L4_DEV_AnnotatorGuidelines** indicator, powered by **OpenAI GPT-4o**, for evaluating whether foundation model developers have formed comprehensive and reusable annotation guidelines and operational instructions for security-related annotation work.

## What is L4_DEV_AnnotatorGuidelines?

L4_DEV_AnnotatorGuidelines belongs to L3 under alignment and training (L2_DEV_AlignTrain) in Model Design and Development (L1_DEV). It evaluates whether AI training is carried out under the guidance and quality inspection of annotators.

### Key Aspects

1. **Written Guidelines Existence**
   - Are there written guidelines or explanatory documents for annotators?
   - Or are instructions only oral or temporary?

2. **Clear Guideline Summaries**
   - Are core value principles explained?
   - Are label meanings clearly defined?
   - How are boundary cases handled?
   - Are typical examples provided?

### Ethical Rationale

Alignment and security annotation essentially tells the model "what is acceptable output". If annotators only get vague and fragmented instructions:

- **Problem**: Different annotators give completely different labels for the same input
- **Result**: The model learns noise and personal preferences, not clear value guidance

**Solution**: Organizing annotation guidelines into structured and reusable instructions makes:
- Alignment targets more transparent and stable
- It easier to evaluate "which set of rules the model has learned"
- The process auditable and accountable

## Scoring Criteria

### ADA (Automated Document Analysis)

| Score | Criteria |
|-------|----------|
| **0** | Only general statements like "use manual feedback for alignment" or "annotate according to safety policies". Cannot see annotation manuals, guideline summaries, or specific examples. Cannot determine if annotators have clear written guidance. |
| **1** | Some annotation-related information visible (mentions "annotator handbook" or provides a few examples and rule fragments). However, overall content is fragmented, lacking systematic summary. Core principles, labeling systems, or boundary situation handling methods are unclear. |
| **2** | Clear summary of annotation guidelines present. Special sections introducing annotation principles, label definitions, prohibited content, gray area handling rules, equipped with typical examples or process descriptions. Shows reusable annotation instructions, not temporary directives. |

### HIJ (Human Interactive Judgment)

| Score | Criteria |
|-------|----------|
| **0** | Existing materials insufficient to show real annotation guidelines exist. Know almost nothing about "how annotators are guided". Cannot consider organization has a clear approach. |
| **1** | Can see certain annotation rules or examples, but content is limited and structure is scattered. Can only prove "there was some guidance", not a relatively complete guideline summary. |
| **2** | Evidence of mutual verification in multiple materials: annotators have clear written guidelines, including principles, label definitions, and typical scenarios. Clear summaries and explanations in documents. Practice has formed a relatively mature system. |

### Final Score Calculation

```
Final Score = ADA Score × 0.5 + HIJ Score × 0.5
```

## Key Features

- ✨ **GPT-4o Powered**: Advanced AI for intelligent annotation guidelines analysis
- 📄 **PDF Support**: Automatic extraction and analysis of PDF documents
- 🔍 **Comprehensive Detection**: Identifies guidelines, manuals, principles, examples, and quality standards
- 🤖 **Automated Scoring**: Hybrid ADA + HIJ evaluation
- 📊 **Detailed Reports**: Complete reports with evidence and reasoning
- 💾 **Multiple Outputs**: Text and JSON format results

## Quick Start

### Installation

```bash
pip install -r ../requirements.txt
```

### Configuration

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Running Evaluation

```bash
cd L4_DEV_AnnotatorGuidelines
python run_annotator_evaluation.py
```

## Evaluation Evidence

GPT-4o analyzes documents to extract evidence for:

| Evidence Type | Description |
|--------------|-------------|
| **has_written_guidelines** | Are there written guidelines for annotators? |
| **has_annotation_manual** | Is there mention of an annotation manual or handbook? |
| **has_core_principles** | Are core value principles described? |
| **has_label_definitions** | Are label categories clearly defined? |
| **has_boundary_case_rules** | Are there rules for handling boundary cases? |
| **has_examples** | Are typical examples provided? |
| **has_quality_standards** | Are quality standards mentioned? |
| **has_process_description** | Is the annotation process described? |
| **has_safety_policy** | Are safety policies mentioned? |
| **guideline_sections_count** | How many guideline sections are covered? |

## Usage Example

```python
from L4_DEV_AnnotatorGuidelines_GPT4o import L4_DEV_AnnotatorGuidelines_GPT4o

# Initialize evaluator
evaluator = L4_DEV_AnnotatorGuidelines_GPT4o()

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
```

## Example Results

Based on the article's analysis, expected scores for evaluated models:

```
===================================================================================
Model                Type            ADA      HIJ      Weighted
-----------------------------------------------------------------------------------
DeepSeek-V3          Open Weights    0        1        0.50
Gemini 2.5           Closed API      1        2        1.50
===================================================================================
```

### Key Findings

- **DeepSeek-V3** (0.50): Only general statements like "using human feedback" and "combined with security strategy annotation". No formed annotation manual or systematic rules. Manual review finds scattered explanations and examples across materials, but not enough to constitute a complete guide.

- **Gemini 2.5** (1.50): More specific explanations of labeling principles, category divisions, and boundaries. Has relatively clear guideline summaries, and manual review confirms a set of reusable labeling rules. Clearer about "what rules annotators work according to".

## Comparison with Other Indicators

This indicator complements data governance indicators by focusing on the annotation process:

| Indicator | Focus | Key Question |
|-----------|-------|--------------|
| **L4_DEV_DatasheetsData** | Dataset documentation | Are datasets documented with version info? |
| **L4_DEV_LineageVersionsMajor** | Data lineage tracking | Can you trace data origins and versions? |
| **L4_DEV_AnnotatorGuidelines** | Annotation guidelines | How are annotators guided in labeling? |

## Theoretical Foundation

1. **Bommasani, R., et al.** (2024). "The 2024 Foundation Model Transparency Index." arXiv:2407.12929
   - Framework for evaluating transparency practices

2. **Dai, J., et al.** (2023). "Safe RLHF: Safe reinforcement learning from human feedback." arXiv:2310.12773
   - Two-dimensional security annotation schemes and detailed guidelines

3. **Zeng, Y., et al.** (2025). "AI Governance International Evaluation Index (AGILE Index)." arXiv:2502.15859
   - AI governance evaluation methodology

4. **Liu, A., et al.** (2024). "Deepseek-v3 technical report." arXiv:2412.19437
   - DeepSeek-V3 technical details

5. **Google DeepMind.** (2025). "Gemini 2.5 Deep Think – Model Card."
   - Gemini 2.5 official documentation

## Project Structure

```
L4_DEV_AnnotatorGuidelines/
├── L4_DEV_AnnotatorGuidelines_GPT4o.py    # Main evaluation tool
├── run_annotator_evaluation.py            # Automated evaluation script
└── README.md                               # This documentation
```

## Output Formats

### Comparison Table

```
===================================================================================
Table: L4_DEV_AnnotatorGuidelines Comparative Assessment (GPT-4o Powered)
===================================================================================
Model                Type            ADA      HIJ      Weighted
-----------------------------------------------------------------------------------
DeepSeek-V3          Open Weights    0        1        0.50
Gemini 2.5           Closed API      1        2        1.50
===================================================================================
```

### Detailed Results

Each evaluation includes:
- ADA and HIJ scores
- Weighted final score
- Guidelines evidence extracted by GPT-4o
- GPT-4o reasoning process
- HIJ review rationale
- Key quotes from documents

### JSON Output

```json
{
  "indicator": "L4_DEV_AnnotatorGuidelines",
  "description": "Annotator guidelines and documentation evaluation",
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
evaluator = L4_DEV_AnnotatorGuidelines_GPT4o()

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
1. Guidelines evidence extraction
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
- How annotation guidelines are presented in documents
- Level of detail in guideline descriptions
- Presence of examples and boundary case rules

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{l4_dev_annotator_guidelines_2025,
  title={L4\_DEV\_AnnotatorGuidelines: Annotation Guidelines Evaluation Tool},
  author={Zhang, Qurui},
  year={2025},
  note={GPT-4o Enhanced Implementation}
}
```

## Related Indicators

This indicator belongs to the AI governance evaluation framework:

- **L1_DEV**: Model Design and Development
  - **L2_DEV_AlignTrain**: Alignment and Training
    - **L3**: Annotator Guidance and Quality Inspection
      - **L4_DEV_AnnotatorGuidelines**: Annotation guidelines documentation (this indicator)

## License

This project is for academic research and educational purposes.

---

**Last Updated**: 2025-12-10
**Version**: 1.0.0 (GPT-4o Enhanced)
