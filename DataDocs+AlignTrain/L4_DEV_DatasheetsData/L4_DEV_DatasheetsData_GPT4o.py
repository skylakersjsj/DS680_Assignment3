"""
L4_DEV_DatasheetsData with OpenAI GPT-4o Integration
====================================================
Enhanced version using OpenAI GPT-4o for automated document analysis and scoring.

This module extends the original implementation by leveraging GPT-4o's advanced
language understanding capabilities for more accurate evidence extraction and scoring.
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
    ABSENT = 0
    PARTIAL = 1
    COMPLETE = 2


@dataclass
class DocumentEvidence:
    """Evidence extracted from documentation"""
    has_datasheet: bool = False
    has_composition_info: bool = False
    has_source_info: bool = False
    has_intended_use: bool = False
    has_limitations: bool = False
    has_version_info: bool = False
    has_timestamp: bool = False
    has_change_records: bool = False
    coverage_ratio: float = 0.0
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
    ada_evidence: DocumentEvidence
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
        """
        Extract text from a PDF file

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to extract (None for all)

        Returns:
            Extracted text
        """
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


class GPT4oDocumentAnalyzer:
    """
    Use OpenAI GPT-4o for intelligent document analysis
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GPT-4o analyzer

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Run: pip install openai")

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"

    def analyze_document_for_evidence(self, document_text: str) -> Tuple[DocumentEvidence, str]:
        """
        Use GPT-4o to analyze document and extract evidence

        Args:
            document_text: Text content to analyze

        Returns:
            Tuple of (DocumentEvidence, reasoning)
        """
        # Truncate if too long (GPT-4o has context limits)
        max_chars = 100000
        if len(document_text) > max_chars:
            document_text = document_text[:max_chars] + "\n\n[Document truncated...]"

        prompt = f"""You are an expert evaluator assessing whether a model developer has established
internal datasheets/data statements for key datasets with version management.

Analyze the following document and determine:

1. **has_datasheet**: Does the document contain or reference datasheets/data statements?
2. **has_composition_info**: Is there information about data composition (what the data contains)?
3. **has_source_info**: Is there information about data sources (where data came from)?
4. **has_intended_use**: Is the intended use or purpose of datasets described?
5. **has_limitations**: Are known limitations or biases of the data mentioned?
6. **has_version_info**: Are there version numbers or version control indicators?
7. **has_timestamp**: Are there dates, timestamps, or time periods mentioned?
8. **has_change_records**: Are there changelogs or records of data changes?
9. **coverage_ratio**: Estimate what proportion (0.0-1.0) of key dataset types (pre-training,
   instruction/fine-tuning, alignment, safety/red-team, evaluation) have documentation.

Also provide 2-3 key quotes that support your assessment.

Document to analyze:
---
{document_text}
---

Respond in JSON format:
{{
    "has_datasheet": true/false,
    "has_composition_info": true/false,
    "has_source_info": true/false,
    "has_intended_use": true/false,
    "has_limitations": true/false,
    "has_version_info": true/false,
    "has_timestamp": true/false,
    "has_change_records": true/false,
    "coverage_ratio": 0.0-1.0,
    "key_quotes": ["quote1", "quote2", "quote3"],
    "reasoning": "Brief explanation of your assessment"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise evaluator of AI model documentation. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            evidence = DocumentEvidence(
                has_datasheet=result.get("has_datasheet", False),
                has_composition_info=result.get("has_composition_info", False),
                has_source_info=result.get("has_source_info", False),
                has_intended_use=result.get("has_intended_use", False),
                has_limitations=result.get("has_limitations", False),
                has_version_info=result.get("has_version_info", False),
                has_timestamp=result.get("has_timestamp", False),
                has_change_records=result.get("has_change_records", False),
                coverage_ratio=result.get("coverage_ratio", 0.0),
                key_quotes=result.get("key_quotes", [])
            )

            reasoning = result.get("reasoning", "")

            return evidence, reasoning

        except Exception as e:
            print(f"Error calling GPT-4o API: {str(e)}")
            # Fallback to empty evidence
            return DocumentEvidence(), f"Error: {str(e)}"

    def calculate_ada_score(self, evidence: DocumentEvidence, reasoning: str) -> Tuple[int, str]:
        """
        Use GPT-4o to calculate ADA score based on evidence

        Args:
            evidence: DocumentEvidence object
            reasoning: Previous reasoning from evidence extraction

        Returns:
            Tuple of (score, detailed_reasoning)
        """
        prompt = f"""Based on the following evidence extracted from model documentation,
assign an ADA (Automated Document Analysis) score according to these criteria:

**ADA = 0**: No datasheet/data statement evidence found; only very general descriptions

**ADA = 1**: At least some explanatory documents of key datasets containing information
about composition, source, or use; but basically static with no clear version number,
date, or change records; incomplete coverage

**ADA = 2**: Key datasets (most or all) have relatively clear datasheets/data statements;
documents have version numbers or time tags; important data changes reflected in versions;
continuous version-based maintenance evident

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
                           evidence: DocumentEvidence,
                           ada_reasoning: str,
                           model_name: str,
                           additional_context: str = "") -> Tuple[int, str]:
        """
        Use GPT-4o to simulate human interactive judgment (HIJ)

        Args:
            ada_score: The ADA score
            evidence: DocumentEvidence object
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

**HIJ = 0**: No credible evidence of internal datasheets/data statements at dataset level,
let alone version management

**HIJ = 1**: Some description of key datasets visible, but coverage incomplete; no clear
version number, timestamp, or change records; can only be counted as "with documents, not versioned"

**HIJ = 2**: Key datasets have clear datasheets/data statements with version or time tags;
important data changes recorded; indicates stable version management practice

As a human reviewer, you may:
- Adjust the score up if you see evidence ADA might have missed
- Adjust the score down if you believe the evidence is insufficient
- Keep the same score if ADA assessment seems accurate

Provide your HIJ score and detailed rationale in JSON format:
{{
    "hij_score": 0/1/2,
    "rationale": "Detailed explanation of your human judgment, including what you observed in the evidence and why you assigned this score"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an experienced human reviewer with deep expertise in AI governance and data documentation practices."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Slightly higher for human-like judgment
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            hij_score = result.get("hij_score", ada_score)
            rationale = result.get("rationale", "")

            return hij_score, rationale

        except Exception as e:
            print(f"Error calculating HIJ score: {str(e)}")
            return ada_score, f"Error in HIJ calculation: {str(e)}. Using ADA score."


class L4_DEV_DatasheetsData_GPT4o:
    """
    Main class using GPT-4o for L4_DEV_DatasheetsData evaluation
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize evaluator with GPT-4o

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        self.analyzer = GPT4oDocumentAnalyzer(api_key)
        self.pdf_extractor = PDFExtractor()
        self.results: List[EvaluationResult] = []

    def load_document(self, path: str, max_pages: int = None) -> str:
        """
        Load document from file (supports PDF and text files)

        Args:
            path: Path to document
            max_pages: For PDFs, maximum pages to extract

        Returns:
            Document text
        """
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
            use_hij: Whether to perform HIJ scoring (vs just ADA)
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
                # Assume it's direct text content
                all_text.append(source)

        combined_text = "\n\n---\n\n".join(all_text)

        # Step 1: Extract evidence with GPT-4o
        print("Step 1: Extracting evidence with GPT-4o...")
        evidence, evidence_reasoning = self.analyzer.analyze_document_for_evidence(combined_text)

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
        table += "Table: L4_DEV_DatasheetsData Comparative Assessment (GPT-4o Powered)\n"
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
        report += "L4_DEV_DatasheetsData - Detailed Evaluation Report (GPT-4o Enhanced)\n"
        report += "="*85 + "\n\n"

        for result in self.results:
            report += str(result) + "\n\n"

        report += "\nCOMPARATIVE SUMMARY\n"
        report += self.generate_comparison_table()

        return report


# Example usage
def main():
    """
    Example: Evaluate DeepSeek-V3 and Gemini 2.5 using actual PDFs
    """
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return

    # Initialize evaluator
    evaluator = L4_DEV_DatasheetsData_GPT4o()

    # Evaluate DeepSeek-V3
    print("\n" + "="*70)
    print("EVALUATING DEEPSEEK-V3")
    print("="*70)

    deepseek_result = evaluator.evaluate_model(
        model_name="DeepSeek-V3",
        model_type="Open Weights",
        document_sources=["references/deepseek_v3.pdf"],
        use_hij=True,
        additional_context="Open-source model emphasizing transparency in model and code, but data governance may not be as mature."
    )

    # Evaluate Gemini 2.5
    print("\n" + "="*70)
    print("EVALUATING GEMINI 2.5")
    print("="*70)

    gemini_result = evaluator.evaluate_model(
        model_name="Gemini 2.5",
        model_type="Closed API",
        document_sources=["references/gemini_2_5_model_card.pdf"],
        use_hij=True,
        additional_context="Large company with established governance processes. Model card format suggests systematic documentation practices."
    )

    # Generate and print reports
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    print(evaluator.generate_detailed_report())

    # Save results to file
    with open("evaluation_results.txt", "w") as f:
        f.write(evaluator.generate_detailed_report())

    print("\n✓ Results saved to: evaluation_results.txt")


if __name__ == "__main__":
    main()
