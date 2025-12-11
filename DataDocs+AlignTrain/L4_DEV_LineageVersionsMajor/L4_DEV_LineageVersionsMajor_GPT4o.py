"""
L4_DEV_LineageVersionsMajor with OpenAI GPT-4o Integration
==========================================================
Implementation of L4_DEV_LineageVersionsMajor indicator for evaluating whether
model developers systematically track the lineage and versions of major datasets.

This indicator measures:
1. Whether major datasets are registered and managed as independent objects with clear
   names/IDs and version labels
2. Whether basic lineage information is recorded (upstream sources, key cleaning/
   filtering steps, major differences between versions)

Evaluation Method: ADA (Automated Document Analysis) + HIJ (Human Interactive Judgment)
Final Score = ADA Score × 0.5 + HIJ Score × 0.5

References:
- Bommasani et al. (2024) - Foundation Model Transparency Index
- González-Cebrián et al. (2024) - Standardised versioning of datasets
- Zeng et al. (2025) - AGILE Index
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Run: pip install openai")

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PyPDF2 not installed. Run: pip install PyPDF2")


class ScoreLevel(Enum):
    """Score levels for ADA and HIJ evaluation"""
    ABSENT = 0      # No lineage/version tracking evidence
    PARTIAL = 1     # Some lineage info, but not systematic
    COMPLETE = 2    # Clear lineage and version tracking


@dataclass
class LineageEvidence:
    """Evidence extracted from documentation about dataset lineage and versioning"""
    has_dataset_names: bool = False
    has_dataset_ids: bool = False
    has_version_labels: bool = False
    has_upstream_sources: bool = False
    has_processing_steps: bool = False
    has_version_differences: bool = False
    has_model_dataset_mapping: bool = False
    has_data_pipeline: bool = False
    named_datasets_count: int = 0
    versioned_datasets_count: int = 0
    key_quotes: List[str] = None

    def __post_init__(self):
        if self.key_quotes is None:
            self.key_quotes = []

    def to_dict(self):
        return asdict(self)


@dataclass
class EvaluationResult:
    """Final evaluation result for a model"""
    model_name: str
    model_type: str
    ada_score: int
    hij_score: int
    weighted_score: float
    ada_evidence: LineageEvidence
    hij_rationale: str
    gpt4o_reasoning: str = ""

    def __str__(self):
        return (f"\n{'='*70}\n"
                f"Model: {self.model_name} ({self.model_type})\n"
                f"{'='*70}\n"
                f"ADA Score (Automated): {self.ada_score}\n"
                f"HIJ Score (Human): {self.hij_score}\n"
                f"Weighted Score: {self.weighted_score:.2f}\n"
                f"\nGPT-4o Reasoning:\n{self.gpt4o_reasoning}\n"
                f"\nHIJ Rationale:\n{self.hij_rationale}\n"
                f"{'='*70}")


class PDFExtractor:
    """Extract text from PDF files"""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str, max_pages: int = None) -> str:
        """Extract text from a PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is required for PDF extraction")

        text = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                if max_pages is not None:
                    num_pages = min(num_pages, max_pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text.append(page.extract_text())

            return "\n".join(text)
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")


class GPT4oLineageAnalyzer:
    """Use OpenAI GPT-4o for intelligent lineage and version analysis"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize GPT-4o analyzer"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Run: pip install openai")

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"

    def analyze_document_for_lineage(self, document_text: str) -> Tuple[LineageEvidence, str]:
        """
        Use GPT-4o to analyze document for dataset lineage and version evidence

        Args:
            document_text: Text content to analyze

        Returns:
            Tuple of (LineageEvidence, reasoning)
        """
        # Truncate if too long
        max_chars = 100000
        if len(document_text) > max_chars:
            document_text = document_text[:max_chars] + "\n\n[Document truncated...]"

        prompt = f"""You are an expert evaluator assessing whether a model developer systematically
tracks the lineage and versions of major datasets (pre-training, instruction/alignment fine-tuning,
safety/red-team evaluation datasets).

Analyze the following document and determine:

1. **has_dataset_names**: Are major datasets given clear names (e.g., "CommonCrawl-2023",
   "InstructionSet-v2")?
2. **has_dataset_ids**: Are datasets identified with specific IDs or identifiers?
3. **has_version_labels**: Are there version labels for datasets (e.g., "v1.0", "version 2",
   "batch 3")?
4. **has_upstream_sources**: Is upstream source information provided (where data came from)?
5. **has_processing_steps**: Are key data cleaning/filtering/processing steps described?
6. **has_version_differences**: Are differences between dataset versions documented?
7. **has_model_dataset_mapping**: Can you see which dataset version corresponds to which
   model version?
8. **has_data_pipeline**: Is there a description of the data pipeline or lineage flow?
9. **named_datasets_count**: How many major datasets are explicitly named (0-10)?
10. **versioned_datasets_count**: How many of these named datasets have version information (0-10)?

Also provide 2-3 key quotes that support your assessment.

Document to analyze:
---
{document_text}
---

Respond in JSON format:
{{
    "has_dataset_names": true/false,
    "has_dataset_ids": true/false,
    "has_version_labels": true/false,
    "has_upstream_sources": true/false,
    "has_processing_steps": true/false,
    "has_version_differences": true/false,
    "has_model_dataset_mapping": true/false,
    "has_data_pipeline": true/false,
    "named_datasets_count": 0-10,
    "versioned_datasets_count": 0-10,
    "key_quotes": ["quote1", "quote2", "quote3"],
    "reasoning": "Brief explanation of your assessment"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise evaluator of dataset lineage and version tracking practices. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            evidence = LineageEvidence(
                has_dataset_names=result.get("has_dataset_names", False),
                has_dataset_ids=result.get("has_dataset_ids", False),
                has_version_labels=result.get("has_version_labels", False),
                has_upstream_sources=result.get("has_upstream_sources", False),
                has_processing_steps=result.get("has_processing_steps", False),
                has_version_differences=result.get("has_version_differences", False),
                has_model_dataset_mapping=result.get("has_model_dataset_mapping", False),
                has_data_pipeline=result.get("has_data_pipeline", False),
                named_datasets_count=result.get("named_datasets_count", 0),
                versioned_datasets_count=result.get("versioned_datasets_count", 0),
                key_quotes=result.get("key_quotes", [])
            )

            reasoning = result.get("reasoning", "")

            return evidence, reasoning

        except Exception as e:
            print(f"Error calling GPT-4o API: {str(e)}")
            return LineageEvidence(), f"Error: {str(e)}"

    def calculate_ada_score(self, evidence: LineageEvidence, reasoning: str) -> Tuple[int, str]:
        """
        Use GPT-4o to calculate ADA score based on lineage evidence

        Args:
            evidence: LineageEvidence object
            reasoning: Previous reasoning from evidence extraction

        Returns:
            Tuple of (score, detailed_reasoning)
        """
        prompt = f"""Based on the following evidence about dataset lineage and version tracking,
assign an ADA (Automated Document Analysis) score according to these criteria:

**ADA = 0**: Only very general statements like "using large-scale internet data" or "mixing
multiple public and authorized data". Cannot see named major datasets, no data pipeline, source
list, or version information. Cannot determine if there is any lineage or version tracking.

**ADA = 1**: Can see some lineage-related information, such as listing several core data sources
or giving brief data processing procedures. However, major datasets do not have clear names or IDs,
cannot see correspondence with specific model versions, no systematic "dataset version table" or
similar structure.

**ADA = 2**: Text contains named/numbered major datasets with their versions or batches mentioned.
At least basic lineage relationships (upstream sources, key cleaning/filtering steps) and major
differences between versions are provided. Can roughly see which dataset version corresponds to
which model versions.

Evidence:
{json.dumps(evidence.to_dict(), indent=2)}

Previous Analysis:
{reasoning}

Respond in JSON format:
{{
    "score": 0/1/2,
    "reasoning": "Detailed explanation of why this score was assigned"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Assign scores precisely according to the rubric."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            score = result.get("score", 0)
            detailed_reasoning = result.get("reasoning", "")

            return score, detailed_reasoning

        except Exception as e:
            print(f"Error calculating ADA score: {str(e)}")
            return 0, f"Error: {str(e)}"

    def calculate_hij_score(self,
                           ada_score: int,
                           evidence: LineageEvidence,
                           ada_reasoning: str,
                           model_name: str,
                           additional_context: str = "") -> Tuple[int, str]:
        """
        Use GPT-4o to simulate human interactive judgment (HIJ)

        Args:
            ada_score: The ADA score
            evidence: LineageEvidence object
            ada_reasoning: Reasoning from ADA
            model_name: Name of the model
            additional_context: Any additional context for judgment

        Returns:
            Tuple of (hij_score, rationale)
        """
        prompt = f"""You are acting as a human expert reviewer conducting HIJ
(Human Interactive Judgment) for model: {model_name}

The automated analysis (ADA) gave a preliminary score of {ada_score}.

Evidence and reasoning:
{json.dumps(evidence.to_dict(), indent=2)}

ADA Reasoning:
{ada_reasoning}

Additional Context:
{additional_context}

HIJ Scoring Rubric:

**HIJ = 0**: Reviewer believes the documentation is basically limited to "vague explanation of
data sources", cannot see reliable lineage or version records of major datasets, cannot indicate
there is a real tracking mechanism.

**HIJ = 1**: Reviewer confirms there is certain lineage information (such as specific descriptions
of sources and processing of some core corpus), but information is fragmented, coverage is limited,
lacks clear version mapping rules. Can only be regarded as "conscious, but not systematic".

**HIJ = 2**: Reviewer sees evidence of mutual verification in multiple materials: major datasets
have fixed names or IDs, version or batch records, key processing steps and correspondence with
model versions are relatively clear. Can be considered to have established relatively stable
lineage and version tracking practices.

As a human reviewer, you may:
- Adjust the score up if you see evidence ADA might have missed
- Adjust the score down if you believe the evidence is insufficient or too scattered
- Keep the same score if ADA assessment seems accurate

Provide your HIJ score and detailed rationale in JSON format:
{{
    "hij_score": 0/1/2,
    "rationale": "Detailed explanation of your human judgment, including what you observed and why you assigned this score"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an experienced human reviewer with deep expertise in data governance and lineage tracking practices."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            hij_score = result.get("hij_score", ada_score)
            rationale = result.get("rationale", "")

            return hij_score, rationale

        except Exception as e:
            print(f"Error calculating HIJ score: {str(e)}")
            return ada_score, f"Error in HIJ calculation: {str(e)}. Using ADA score."


class L4_DEV_LineageVersionsMajor_GPT4o:
    """
    Main class using GPT-4o for L4_DEV_LineageVersionsMajor evaluation
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize evaluator with GPT-4o"""
        self.analyzer = GPT4oLineageAnalyzer(api_key)
        self.pdf_extractor = PDFExtractor()
        self.results: List[EvaluationResult] = []

    def load_document(self, path: str, max_pages: int = None) -> str:
        """Load document from file (supports PDF and text files)"""
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if path_obj.suffix.lower() == '.pdf':
            return self.pdf_extractor.extract_text_from_pdf(str(path_obj), max_pages)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()

    def evaluate_model(self,
                      model_name: str,
                      model_type: str,
                      document_sources: List[str],
                      use_hij: bool = True,
                      additional_context: str = "") -> EvaluationResult:
        """
        Evaluate a model using GPT-4o

        Args:
            model_name: Name of the model
            model_type: Type ("Open Weights" or "Closed API")
            document_sources: List of file paths or text content
            use_hij: Whether to perform HIJ scoring
            additional_context: Additional context for HIJ

        Returns:
            EvaluationResult
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}\n")

        # Load and combine documents
        all_text = []
        for source in document_sources:
            if os.path.isfile(source):
                print(f"Loading document: {source}")
                text = self.load_document(source)
                all_text.append(text)
            else:
                all_text.append(source)

        combined_text = "\n\n---\n\n".join(all_text)

        # Step 1: Extract lineage evidence with GPT-4o
        print("Step 1: Extracting lineage evidence with GPT-4o...")
        evidence, evidence_reasoning = self.analyzer.analyze_document_for_lineage(combined_text)

        # Step 2: Calculate ADA score
        print("Step 2: Calculating ADA score...")
        ada_score, ada_reasoning = self.analyzer.calculate_ada_score(evidence, evidence_reasoning)
        print(f"  ADA Score: {ada_score}")

        # Step 3: Calculate HIJ score (if enabled)
        if use_hij:
            print("Step 3: Calculating HIJ score...")
            hij_score, hij_rationale = self.analyzer.calculate_hij_score(
                ada_score, evidence, ada_reasoning, model_name, additional_context
            )
            print(f"  HIJ Score: {hij_score}")
        else:
            hij_score = ada_score
            hij_rationale = "HIJ not performed, using ADA score."

        # Step 4: Calculate weighted score
        weighted_score = ada_score * 0.5 + hij_score * 0.5

        # Combine all reasoning
        full_reasoning = f"""Evidence Analysis:
{evidence_reasoning}

ADA Scoring:
{ada_reasoning}"""

        result = EvaluationResult(
            model_name=model_name,
            model_type=model_type,
            ada_score=ada_score,
            hij_score=hij_score,
            weighted_score=weighted_score,
            ada_evidence=evidence,
            hij_rationale=hij_rationale,
            gpt4o_reasoning=full_reasoning
        )

        self.results.append(result)
        print(f"\nFinal Weighted Score: {weighted_score:.2f}\n")

        return result

    def generate_comparison_table(self) -> str:
        """Generate comparison table"""
        if not self.results:
            return "No evaluation results available."

        table = "\n" + "="*85 + "\n"
        table += "Table: L4_DEV_LineageVersionsMajor Comparative Assessment (GPT-4o Powered)\n"
        table += "="*85 + "\n"
        table += f"{'Model':<20} {'Type':<15} {'ADA':<8} {'HIJ':<8} {'Weighted':<10}\n"
        table += "-"*85 + "\n"

        for result in self.results:
            table += f"{result.model_name:<20} {result.model_type:<15} "
            table += f"{result.ada_score:<8} {result.hij_score:<8} "
            table += f"{result.weighted_score:<10.2f}\n"

        table += "="*85 + "\n"
        return table

    def generate_detailed_report(self) -> str:
        """Generate detailed report"""
        report = "\n" + "="*85 + "\n"
        report += "L4_DEV_LineageVersionsMajor - Detailed Evaluation Report (GPT-4o Enhanced)\n"
        report += "="*85 + "\n\n"

        for result in self.results:
            report += str(result) + "\n\n"

        report += "\nCOMPARATIVE SUMMARY\n"
        report += self.generate_comparison_table()

        return report


# Example usage
def main():
    """
    Example: Evaluate DeepSeek-V3 and Gemini 2.5 for lineage and version tracking
    """
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return

    # Initialize evaluator
    evaluator = L4_DEV_LineageVersionsMajor_GPT4o()

    # Evaluate DeepSeek-V3
    print("\n" + "="*70)
    print("EVALUATING DEEPSEEK-V3")
    print("="*70)

    deepseek_result = evaluator.evaluate_model(
        model_name="DeepSeek-V3",
        model_type="Open Weights",
        document_sources=["references/deepseek_v3.pdf"],
        use_hij=True,
        additional_context=(
            "Open-source model emphasizing transparency in model and code. "
            "Data-level lineage and version governance may be less mature than "
            "model/code transparency."
        )
    )

    # Evaluate Gemini 2.5
    print("\n" + "="*70)
    print("EVALUATING GEMINI 2.5")
    print("="*70)

    gemini_result = evaluator.evaluate_model(
        model_name="Gemini 2.5 Deep Think",
        model_type="Closed API",
        document_sources=["references/gemini_2_5_model_card.pdf"],
        use_hij=True,
        additional_context=(
            "Large company with established governance processes. Model card format "
            "suggests more structured documentation of data sources and processing. "
            "May have internal lineage tracking not fully disclosed in public documents."
        )
    )

    # Generate and print reports
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    print(evaluator.generate_detailed_report())

    # Save results to file
    with open("lineage_evaluation_results.txt", "w") as f:
        f.write(evaluator.generate_detailed_report())

    print("\n✓ Results saved to: lineage_evaluation_results.txt")

    # Save JSON results
    import json
    json_output = {
        "indicator": "L4_DEV_LineageVersionsMajor",
        "description": "Dataset lineage and version tracking evaluation",
        "models": [
            {
                "name": deepseek_result.model_name,
                "type": deepseek_result.model_type,
                "ada_score": deepseek_result.ada_score,
                "hij_score": deepseek_result.hij_score,
                "weighted_score": deepseek_result.weighted_score,
                "evidence": deepseek_result.ada_evidence.to_dict(),
                "hij_rationale": deepseek_result.hij_rationale,
                "gpt4o_reasoning": deepseek_result.gpt4o_reasoning
            },
            {
                "name": gemini_result.model_name,
                "type": gemini_result.model_type,
                "ada_score": gemini_result.ada_score,
                "hij_score": gemini_result.hij_score,
                "weighted_score": gemini_result.weighted_score,
                "evidence": gemini_result.ada_evidence.to_dict(),
                "hij_rationale": gemini_result.hij_rationale,
                "gpt4o_reasoning": gemini_result.gpt4o_reasoning
            }
        ]
    }

    json_file = "lineage_evaluation_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"✓ JSON results saved to: {json_file}")


if __name__ == "__main__":
    main()
