#!/usr/bin/env python3
"""
Quick Start Script for L4_DEV_AnnotatorGuidelines Evaluation with GPT-4o

This script evaluates whether model developers have formed comprehensive and
reusable annotation guidelines for security-related annotation work.
"""

import os
import sys
from pathlib import Path


def check_api_key():
    """Check if OpenAI API key is configured"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("="*70)
        print("ERROR: OpenAI API key not found!")
        print("="*70)
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://platform.openai.com/api-keys")
        print("="*70)
        sys.exit(1)
    return api_key


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import openai
        import PyPDF2
        return True
    except ImportError as e:
        print("="*70)
        print("ERROR: Missing required dependencies!")
        print("="*70)
        print(f"\nMissing package: {e.name}")
        print("\nPlease install required packages:")
        print("  pip install -r ../requirements.txt")
        print("="*70)
        sys.exit(1)


def main():
    """Main evaluation script"""
    print("\n" + "="*70)
    print("L4_DEV_AnnotatorGuidelines Evaluation Tool (GPT-4o Enhanced)")
    print("="*70 + "\n")

    # Check prerequisites
    print("Checking prerequisites...")
    check_dependencies()
    api_key = check_api_key()
    print("✓ All prerequisites met\n")

    # Import the evaluator
    from L4_DEV_AnnotatorGuidelines_GPT4o import L4_DEV_AnnotatorGuidelines_GPT4o

    # Initialize evaluator
    evaluator = L4_DEV_AnnotatorGuidelines_GPT4o(api_key=api_key)

    # Check if reference documents exist
    candidate_dirs = [Path("../references"), Path("references")]  # allow running from root/subdir
    ref_dir = None
    for candidate in candidate_dirs:
        deepseek_pdf = candidate / "deepseek_v3.pdf"
        gemini_pdf = candidate / "gemini_2_5_model_card.pdf"
        if deepseek_pdf.exists() and gemini_pdf.exists():
            ref_dir = candidate
            break

    if ref_dir is None:
        ref_dir = Path("../references")
        deepseek_pdf = ref_dir / "deepseek_v3.pdf"
        gemini_pdf = ref_dir / "gemini_2_5_model_card.pdf"
        print("WARNING: Reference PDFs not found in references/ or ../references/")
        print("Expected files:")
        print(f"  - {deepseek_pdf}")
        print(f"  - {gemini_pdf}")
        print("\nPlease ensure reference documents are available.")
        sys.exit(1)

    deepseek_pdf = ref_dir / "deepseek_v3.pdf"
    gemini_pdf = ref_dir / "gemini_2_5_model_card.pdf"

    print(f"✓ Found reference documents\n")

    # Evaluate DeepSeek-V3
    print("="*70)
    print("STEP 1/2: Evaluating DeepSeek-V3")
    print("="*70)
    print("Analyzing annotator guidelines documentation...\n")

    deepseek_result = evaluator.evaluate_model(
        model_name="DeepSeek-V3",
        model_type="Open Weights",
        document_sources=[str(deepseek_pdf)],
        use_hij=True,
        additional_context=(
            "Open-source model from DeepSeek. Technical report publicly available. "
            "May contain general alignment and RLHF descriptions, but detailed "
            "annotator guidelines are often not publicly disclosed in open-source models."
        )
    )

    # Evaluate Gemini 2.5
    print("\n" + "="*70)
    print("STEP 2/2: Evaluating Gemini 2.5")
    print("="*70)
    print("Analyzing annotator guidelines documentation...\n")

    gemini_result = evaluator.evaluate_model(
        model_name="Gemini 2.5 Deep Think",
        model_type="Closed API",
        document_sources=[str(gemini_pdf)],
        use_hij=True,
        additional_context=(
            "Closed-source model from Google DeepMind. Model card documentation. "
            "Large tech companies typically have more established annotation processes "
            "and may include more detailed guidelines about safety categories, "
            "annotation principles, and quality standards in their documentation."
        )
    )

    # Generate and display results
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    print(evaluator.generate_comparison_table())

    # Display detailed results
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)

    print(deepseek_result)
    print(gemini_result)

    # Save results to file
    output_file = "annotator_guidelines_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(evaluator.generate_detailed_report())

    print(f"\n✓ Detailed results saved to: {output_file}")

    # Save JSON results
    import json
    json_output = {
        "indicator": "L4_DEV_AnnotatorGuidelines",
        "description": "Annotator guidelines and documentation evaluation",
        "evaluation_date": "2025-12-10",
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

    json_file = "annotator_guidelines_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"✓ JSON results saved to: {json_file}")

    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"DeepSeek-V3:         ADA={deepseek_result.ada_score}, HIJ={deepseek_result.hij_score}, Final={deepseek_result.weighted_score:.2f}")
    print(f"Gemini 2.5:          ADA={gemini_result.ada_score}, HIJ={gemini_result.hij_score}, Final={gemini_result.weighted_score:.2f}")
    print("="*70)
    print("\nExpected scores (from article):")
    print("  DeepSeek-V3: ADA=0, HIJ=1, Final=0.5")
    print("  Gemini 2.5:  ADA=1, HIJ=2, Final=1.5")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
