#!/usr/bin/env python3
"""Quick start script for L4_DEV_EduGoalSpec indicator."""

import os
import sys
from pathlib import Path
import json


def ensure_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("=" * 70)
        print("ERROR: OpenAI API key not set!")
        print("=" * 70)
        print("Run: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    return api_key


def ensure_dependencies() -> None:
    try:
        import openai  # noqa: F401
        import PyPDF2  # noqa: F401
    except ImportError as exc:
        print("=" * 70)
        print("ERROR: Missing dependency")
        print("=" * 70)
        print(f"Missing package: {exc.name}")
        print("Install shared requirements with: pip install -r ../requirements.txt")
        sys.exit(1)


def main():
    print("\n" + "=" * 70)
    print("L4_DEV_EduGoalSpec Evaluation Tool (GPT-4o)")
    print("=" * 70 + "\n")

    ensure_dependencies()
    api_key = ensure_api_key()
    print("✓ Environment ready\n")

    from L4_DEV_EduGoalSpec_GPT4o import L4_DEV_EduGoalSpec_GPT4o

    evaluator = L4_DEV_EduGoalSpec_GPT4o(api_key=api_key)

    candidate_dirs = [Path("../references"), Path("references")]  # support both cwd patterns
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
        print("ERROR: Missing reference PDFs in references/ or ../references/")
        print(f"Expected: {deepseek_pdf} and {gemini_pdf}")
        sys.exit(1)

    deepseek_pdf = ref_dir / "deepseek_v3.pdf"
    gemini_pdf = ref_dir / "gemini_2_5_model_card.pdf"

    print("✓ Reference docs found\n")

    print("=" * 70)
    print("STEP 1/2: DeepSeek-V3")
    print("=" * 70)
    deepseek_result = evaluator.evaluate_model(
        model_name="DeepSeek-V3",
        model_type="Open Weights",
        document_sources=[str(deepseek_pdf)],
        use_hij=True,
        additional_context=(
            "Technical report mostly provides responsible-use tips; rarely discloses measurable education goals or KPIs."
        ),
    )

    print("=" * 70)
    print("STEP 2/2: Gemini 2.5")
    print("=" * 70)
    gemini_result = evaluator.evaluate_model(
        model_name="Gemini 2.5",
        model_type="Closed API",
        document_sources=[str(gemini_pdf)],
        use_hij=True,
        additional_context=(
            "Model card mentions user education importance and some measurement dimensions, yet lacks full goal+KPI+baseline design."
        ),
    )

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(evaluator.generate_comparison_table())

    detailed_report = evaluator.generate_detailed_report()
    with open("edu_goal_spec_results.txt", "w", encoding="utf-8") as handle:
        handle.write(detailed_report)
    print("✓ Saved text report to edu_goal_spec_results.txt")

    json_output = {
        "indicator": "L4_DEV_EduGoalSpec",
        "description": "Education goal specification evaluation",
        "models": [
            {
                "name": deepseek_result.model_name,
                "type": deepseek_result.model_type,
                "ada_score": deepseek_result.ada_score,
                "hij_score": deepseek_result.hij_score,
                "weighted_score": deepseek_result.weighted_score,
                "evidence": deepseek_result.ada_evidence.to_dict(),
                "hij_rationale": deepseek_result.hij_rationale,
                "gpt4o_reasoning": deepseek_result.gpt4o_reasoning,
            },
            {
                "name": gemini_result.model_name,
                "type": gemini_result.model_type,
                "ada_score": gemini_result.ada_score,
                "hij_score": gemini_result.hij_score,
                "weighted_score": gemini_result.weighted_score,
                "evidence": gemini_result.ada_evidence.to_dict(),
                "hij_rationale": gemini_result.hij_rationale,
                "gpt4o_reasoning": gemini_result.gpt4o_reasoning,
            },
        ],
    }

    with open("edu_goal_spec_results.json", "w", encoding="utf-8") as handle:
        json.dump(json_output, handle, indent=2, ensure_ascii=False)
    print("✓ Saved JSON report to edu_goal_spec_results.json")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        f"DeepSeek-V3 → ADA={deepseek_result.ada_score}, HIJ={deepseek_result.hij_score}, Final={deepseek_result.weighted_score:.2f}"
    )
    print(
        f"Gemini 2.5  → ADA={gemini_result.ada_score}, HIJ={gemini_result.hij_score}, Final={gemini_result.weighted_score:.2f}"
    )
    print("Expected reference scores: DeepSeek-V3=0.00, Gemini 2.5=0.50")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
