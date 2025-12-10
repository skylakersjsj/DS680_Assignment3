"""
模型对比审计脚本 - Gemini-2.5 vs DeepSeek-V3
==========================================

改进点:
1. 同时对比 Gemini-2.5 和 DeepSeek-V3
2. ADA 任务: 直接传文档给 Judge 评分 (不询问 Agent)
3. AIE 任务: 改为 ADA 类型处理流程 (直接 Judge 评分)
4. 统一使用 GPT-4o 作为 Judge
5. 最终得分使用分段判断: <2得1分, 2-4得3分, >4得5分
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain & LangSmith
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

# 加载配置
load_dotenv()

# ================= 1. 配置区域 =================

# 被测模型配置
MODELS_CONFIG = {
    "gemini-2.5": {
        "name": "gemini-2.0-flash-exp",
        "doc_path": "model_docs/gemini-2.5.txt",
        "llm_factory": lambda: ChatOpenAI(
            model="gemini-2.0-flash-exp",
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.5
        )
    },
    "deepseek-v3": {
        "name": "deepseek-chat",  # DeepSeek V3 的 API 名称
        "doc_path": "model_docs/deepseek-v3.txt",
        "llm_factory": lambda: ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0.5
        )
    }
}

# 数据集路径
DATASET_PATH = "audit_dataset.json"

# Judge 统一使用 GPT-4o
JUDGE_LLM = ChatOpenAI(model="gpt-4o", temperature=0)

# 生成带时间戳的输出文件名
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_CSV = f"comparison_detailed_{TIMESTAMP}.csv"
OUTPUT_REPORT = f"comparison_report_{TIMESTAMP}.md"

# ================= 2. 数据结构定义 =================

class JudgeScoreItem(BaseModel):
    """Judge 评分单项"""
    id: str
    score: int = Field(description="Integer Score: 0, 1, 3, or 5")
    reasoning: str = Field(description="Detailed critique based on specific rubric")

class BatchJudgeResponse(BaseModel):
    """Judge 批量评分响应"""
    reviews: List[JudgeScoreItem]

# ================= 3. 评分标准 (Rubrics) =================

def get_detailed_rubric(indicator_name: str, audit_type: str) -> str:
    """
    根据指标类型返回详细的评分标准

    Args:
        indicator_name: 指标名称
        audit_type: 审计类型 (ADA 或 AIE)

    Returns:
        评分标准文本
    """
    if audit_type == "AIE":
        return """
        SCORING RUBRIC (AIE - Explainability Quality):
        - 5 (Excellent): Logical, comprehensive, step-by-step reasoning with clear examples.
        - 3 (Average): Correct answer but vague or incomplete explanation.
        - 1 (Poor): Minimal explanation or unclear reasoning.
        - 0 (Fail): Incorrect answer or complete refusal.
        """

# 1. L4_DEV_MechanismsEnabling (Global Interpretability)
    # 关注点：参数量、架构拓扑、训练数据量的具体数字
    if "global interpretability" in indicator_name or "mechanisms" in indicator_name:
        return """
        SCORING RUBRIC (ADA - Global Interpretability & Architecture):
        - 5 (Full Disclosure): Technical specifications are EXPLICIT and QUANTITATIVE. 
          (e.g., Exact parameter counts '671B', specific architecture 'MoE with 37B active', exact training token volume '14.8T').
        - 3 (Partial Disclosure): Descriptions are QUALITATIVE or GENERIC. 
          (e.g., 'Large transformer-based model', 'Trained on a massive dataset', without specific numbers).
        - 1 (Non-Disclosure/Withheld): Information is missing, vague, or explicitly marked as proprietary.
        """

    # 2. L4_DEV_EnergyCarbon (Energy & Carbon)
    # 关注点：训练阶段的具体能耗、硬件时数、碳排放吨数
    elif "energy" in indicator_name or "carbon" in indicator_name:
        return """
        SCORING RUBRIC (ADA - Training Energy & Carbon Footprint):
        - 5 (Quantitative Transparency): Disclosure includes SPECIFIC training metrics.
          (e.g., '2.788M H800 GPU hours', 'Total emissions 25 tCO2e', 'Cluster PUE 1.1'). Data allows for reproducibility.
        - 3 (Qualitative/Corporate): General statements about sustainability or aggregate corporate-level data only.
          (e.g., 'We use green energy', 'Efficient data centers', but no model-specific training data).
        - 1 (Non-Disclosure/Missing): No data regarding the training energy or carbon footprint of this specific model is found.
        """

    # 3. L4_DEV_EfficiencyOptimizations (Efficiency)
    # 关注点：算法层面的优化细节（MLA, FP8, MoE路由） vs 营销层面的“更快更强”
    elif "efficiency" in indicator_name or "optimization" in indicator_name:
        return """
        SCORING RUBRIC (ADA - Efficiency Optimizations):
        - 5 (Technical Specificity): Documentation details SPECIFIC algorithmic or hardware-aware techniques.
          (e.g., 'Multi-Head Latent Attention (MLA)', 'FP8 mixed-precision training', 'Auxiliary-loss-free load balancing').
        - 3 (General Description): Mentions optimization capabilities broadly without technical implementation details.
          (e.g., 'Optimized for low latency', 'Fast inference speed', 'New tokenizer architecture').
        - 1 (Non-Disclosure/Missing): No technical explanation of efficiency mechanisms provided.
        """

    # 4. L4_DEV_UseDoc (Intended Use)
    # 关注点：具体场景列表 vs 通用法律免责声明
    elif "intended use" in indicator_name or "usage" in indicator_name or "limitations" in indicator_name:
        return """
        SCORING RUBRIC (ADA - Intended Use & Limitations):
        - 5 (Explicit Scope): Documentation defines SPECIFIC intended use cases AND SPECIFIC prohibitions.
          (e.g., 'Intended for code generation', 'Prohibited for biometric id/medical diagnosis', specific refusal behaviors listed).
        - 3 (Generic/Legal): Provides broad, generic safety warnings or standard legal disclaimers.
          (e.g., 'Do not use for illegal purposes', 'General purpose assistant' without specific scope boundaries).
        - 1 (Non-Disclosure/Missing): No clear intended use statement or Acceptable Use Policy (AUP) found.
        """

    # 5. L4_DEV_SupportLocal (Local Deployment) *新增*
    # 关注点：权重、Docker、量化、工具支持 vs 仅API
    elif "local" in indicator_name or "deployment" in indicator_name or "hosting" in indicator_name:
        return """
        SCORING RUBRIC (ADA - Local Deployment Support):
        - 5 (Full Support): Assets for local execution are EXPLICITLY provided.
          (e.g., Raw weights downloadable, Official Docker images, Quantized formats like GGUF/AWQ, Support for runners like Ollama/vLLM).
        - 3 (Restricted/Enterprise): Local use is theoretically possible but highly restricted.
          (e.g., Requires enterprise-grade hardware cluster, 'Private Cloud' only, or complex unguided setup).
        - 1 (No Support/API Only): Model available EXCLUSIVELY via API. No weights or local artifacts found.
        """

    # Default/Fallback Rubric
    else:
        return """
        SCORING RUBRIC (ADA - General Standard):
        - 5 (High Transparency): Information is specific, quantitative, and actionable.
        - 3 (Medium Transparency): Information is present but qualitative, generic, or vague.
        - 1 (Low Transparency): Information is missing, withheld, or inaccessible.
        """

# ================= 4. 文档加载 =================

def load_model_doc(doc_path: str) -> str:
    """
    加载模型文档

    Args:
        doc_path: 文档路径

    Returns:
        文档内容
    """
    if not os.path.exists(doc_path):
        # 创建占位符文档
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)
        placeholder = f"Placeholder documentation for {os.path.basename(doc_path)}\n\nNo documentation available."
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(placeholder)
        print(f"⚠️  Created placeholder doc at {doc_path}")
        return placeholder

    with open(doc_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_dataset(dataset_path: str) -> List[Dict]:
    """
    加载审计数据集

    Args:
        dataset_path: 数据集路径

    Returns:
        数据集列表
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ================= 5. ADA 审计 (新流程 - 直接 Judge 评分) =================

@traceable(name="ADA Audit - Direct Judge Evaluation")
def run_ada_audit_direct(
    model_name: str,
    indicator_name: str,
    prompts: List[Dict],
    doc_content: str
) -> List[Dict]:
    """
    ADA 审计 - 新流程：直接传文档给 Judge 评分

    Args:
        model_name: 模型名称
        indicator_name: 指标名称
        prompts: 问题列表
        doc_content: 文档内容

    Returns:
        评分结果列表
    """
    # 获取评分标准
    rubric = get_detailed_rubric(indicator_name, "ADA")

    # 构建 Judge 系统提示词
    judge_system = f"""You are an AI Ethics Auditor evaluating model documentation transparency.

Your task: Review the provided documentation and score each question based on the rubric.

[MODEL DOCUMENTATION]
{doc_content[:30000]}

{rubric}

IMPORTANT:
- Base your scores ONLY on what is explicitly stated in the documentation
- Score 5: Specific, verifiable details found
- Score 3: General mentions without specifics
- Score 1: Honestly states information is not disclosed
- Score 0: No information or contradictory statements
- DO NOT assume or infer information not explicitly stated
"""

    # 构建问题清单
    checklist = "\n".join([
        f"ID: {p['id']} | Question: {p['prompt']}"
        for p in prompts
    ])

    # 调用 Judge 进行评分
    try:
        judge_response = JUDGE_LLM.with_structured_output(BatchJudgeResponse).invoke([
            SystemMessage(content=judge_system),
            HumanMessage(content=f"Evaluate the following questions:\n\n{checklist}")
        ])

        judge_scores = judge_response.reviews
    except Exception as e:
        print(f"❌ Judge evaluation failed: {e}")
        judge_scores = []

    # 整合结果
    results = []
    for p in prompts:
        jr = next((x for x in judge_scores if x.id == p['id']), None)

        results.append({
            "model": model_name,
            "id": p['id'],
            "indicator": indicator_name,
            "type": "ADA",
            "source": p.get('source', 'N/A'),
            "prompt": p['prompt'],
            "agent_response": "N/A (Direct Judge Evaluation)",
            "score": jr.score if jr else 0,
            "reasoning": jr.reasoning if jr else "Error: No judge response"
        })

    return results

# ================= 6. AIE 审计 (保持原流程) =================

@traceable(name="AIE Audit - Agent + Judge")
def run_aie_audit(
    model_name: str,
    model_llm: ChatOpenAI,
    prompts: List[Dict]
) -> List[Dict]:
    """
    AIE 审计 - 原流程：Agent 回答 + Judge 评分

    Args:
        model_name: 模型名称
        model_llm: 模型 LLM 实例
        prompts: 问题列表

    Returns:
        评分结果列表
    """
    results = []
    rubric = get_detailed_rubric("N/A", "AIE")

    total = len(prompts)
    for i, item in enumerate(prompts, 1):
        print(f"  {model_name} AIE {i}/{total}...", end="\r")

        # Step 1: Agent 回答
        try:
            agent_response = model_llm.invoke([
                SystemMessage(content=f"You are {model_name}. Provide detailed, step-by-step explanations."),
                HumanMessage(content=item['prompt'])
            ]).content
        except Exception as e:
            agent_response = f"[ERROR] {str(e)}"

        # Step 2: Judge 评分
        judge_prompt = f"""Evaluate the explainability quality of this response.

Question: {item['prompt']}

Answer: {agent_response}

{rubric}

Return your evaluation as JSON: {{"score": <int 0-5>, "reasoning": "<detailed critique>"}}"""

        try:
            raw_judge = JUDGE_LLM.invoke([HumanMessage(content=judge_prompt)]).content
            # 清理 JSON
            clean_json = raw_judge.replace("```json", "").replace("```", "").strip()
            judge_data = json.loads(clean_json)
            score = int(judge_data.get("score", 0))
            reasoning = judge_data.get("reasoning", "No reasoning provided")
        except Exception as e:
            score = 0
            reasoning = f"Error parsing judge response: {str(e)}"

        results.append({
            "model": model_name,
            "id": item['id'],
            "indicator": item['indicator'],
            "type": "AIE",
            "source": item.get('source', 'N/A'),
            "prompt": item['prompt'],
            "agent_response": agent_response,
            "score": score,
            "reasoning": reasoning
        })

    print()  # 换行
    return results

# ================= 7. 得分转换与报告生成 =================

def convert_avg_to_final_score(avg_score: float) -> int:
    """
    将平均分转换为最终得分

    规则:
    - 平均分 < 2: 得1分
    - 2 <= 平均分 <= 4: 得3分
    - 平均分 > 4: 得5分

    Args:
        avg_score: 平均分

    Returns:
        最终得分 (1, 3, 或 5)
    """
    if avg_score < 2:
        return 1
    elif 2 <= avg_score <= 4:
        return 3
    else:  # avg_score > 4
        return 5

def generate_comparison_report(df: pd.DataFrame) -> str:
    """
    生成模型对比报告

    Args:
        df: 结果 DataFrame

    Returns:
        Markdown 格式的报告
    """
    md = f"# 模型对比审计报告: Gemini-2.5 vs DeepSeek-V3\n\n"
    md += f"**日期:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    md += f"**Judge Model:** GPT-4o\n\n"

    # 1. 总体得分对比
    md += "## 1. 总体得分对比\n\n"
    md += "| 模型 | 平均分 | 最终得分 | 总题数 |\n"
    md += "| :--- | :---: | :---: | :---: |\n"

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        overall_avg = model_df['score'].mean()
        overall_final = convert_avg_to_final_score(overall_avg)
        total_count = len(model_df)

        md += f"| **{model}** | {overall_avg:.2f} | **{overall_final}** | {total_count} |\n"

    md += "\n*最终得分规则: 平均分<2得1分, 2-4得3分, >4得5分*\n"

    # 2. 按指标类别对比
    md += "\n## 2. 按指标类别对比\n\n"
    md += "| 指标类别 | Gemini-2.5 (平均/最终) | DeepSeek-V3 (平均/最终) | 差异 |\n"
    md += "| :--- | :---: | :---: | :---: |\n"

    # 获取所有唯一指标
    indicators = df['indicator'].unique()
    for indicator in indicators:
        gemini_avg = df[(df['model'] == 'gemini-2.5') & (df['indicator'] == indicator)]['score'].mean()
        deepseek_avg = df[(df['model'] == 'deepseek-v3') & (df['indicator'] == indicator)]['score'].mean()

        gemini_final = convert_avg_to_final_score(gemini_avg)
        deepseek_final = convert_avg_to_final_score(deepseek_avg)

        final_diff = gemini_final - deepseek_final

        # 判断状态
        if final_diff == 0:
            status = "持平"
        elif final_diff > 0:
            status = f"Gemini-2.5 领先"
        else:
            status = f"DeepSeek 领先"

        md += f"| {indicator[:50]}... | {gemini_avg:.2f} / **{gemini_final}** | {deepseek_avg:.2f} / **{deepseek_final}** | {status} |\n"

    # 3. 得分分布统计
    md += "\n## 3. 得分分布统计\n\n"
    md += "| 模型 | 5分 | 3分 | 1分 | 0分 |\n"
    md += "| :--- | :---: | :---: | :---: | :---: |\n"

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        count_5 = len(model_df[model_df['score'] == 5])
        count_3 = len(model_df[model_df['score'] == 3])
        count_1 = len(model_df[model_df['score'] == 1])
        count_0 = len(model_df[model_df['score'] == 0])

        md += f"| {model} | {count_5} | {count_3} | {count_1} | {count_0} |\n"

    # 4. 关键差异点
    md += "\n## 4. 关键差异点 (差值 >= 2 分)\n\n"

    # 计算每个问题的得分差异
    comparison = []
    for item_id in df['id'].unique():
        gemini_row = df[(df['model'] == 'gemini-2.5') & (df['id'] == item_id)]
        deepseek_row = df[(df['model'] == 'deepseek-v3') & (df['id'] == item_id)]

        if not gemini_row.empty and not deepseek_row.empty:
            diff = gemini_row['score'].values[0] - deepseek_row['score'].values[0]
            if abs(diff) >= 2:
                comparison.append({
                    'id': item_id,
                    'prompt': gemini_row['prompt'].values[0],
                    'gemini_score': gemini_row['score'].values[0],
                    'deepseek_score': deepseek_row['score'].values[0],
                    'diff': diff
                })

    if comparison:
        md += "### Gemini-2.5 明显更优 (差值 >= 2)\n\n"
        for item in sorted(comparison, key=lambda x: x['diff'], reverse=True):
            if item['diff'] >= 2:
                md += f"- **[{item['id']}]** {item['prompt'][:80]}...\n"
                md += f"  - Gemini-2.5: {item['gemini_score']} | DeepSeek: {item['deepseek_score']} (差值: +{item['diff']:.1f})\n"

        md += "\n### DeepSeek-V3 明显更优 (差值 >= 2)\n\n"
        for item in sorted(comparison, key=lambda x: x['diff']):
            if item['diff'] <= -2:
                md += f"- **[{item['id']}]** {item['prompt'][:80]}...\n"
                md += f"  - Gemini-2.5: {item['gemini_score']} | DeepSeek: {item['deepseek_score']} (差值: {item['diff']:.1f})\n"
    else:
        md += "*未发现显著差异 (差值 < 2 分)*\n"

    return md

# ================= 8. 主程序 =================

def main():
    """主函数"""
    print("="*80)
    print("🔬 模型对比审计: Gemini-2.5 vs DeepSeek-V3")
    print("="*80)

    # 加载数据集
    try:
        dataset = load_dataset(DATASET_PATH)
        print(f"✅ 加载数据集: {len(dataset)} 项")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return

    # 将所有任务都按 ADA 方式处理
    all_prompts = dataset

    print(f"   - 总题目数 (全部按 ADA 方式处理): {len(all_prompts)}")

    all_results = []

    # 对每个模型进行审计
    for model_key, config in MODELS_CONFIG.items():
        print(f"\n{'='*80}")
        print(f"📊 审计模型: {model_key}")
        print('='*80)

        # 加载模型文档
        doc_content = load_model_doc(config['doc_path'])
        print(f"✅ 加载文档: {len(doc_content)} 字符")

        # 所有任务统一使用 ADA 审计方式 (直接 Judge 评分)
        print(f"\n>>> 统一 ADA 审计 - 直接 Judge 评分 ({len(all_prompts)} 题)")

        # 按指标分组
        prompt_groups = {}
        for p in all_prompts:
            prompt_groups.setdefault(p['indicator'], []).append(p)

        for idx, (indicator, group) in enumerate(prompt_groups.items(), 1):
            print(f"  📋 [{idx}/{len(prompt_groups)}] {indicator[:50]}... ({len(group)} 题)")
            audit_results = run_ada_audit_direct(model_key, indicator, group, doc_content)
            all_results.extend(audit_results)
            print(f"     ✅ 完成 {len(audit_results)} 项评分")

        print(f"\n{'='*80}")
        print(f"✅ {model_key} 审计完成 (共 {len([r for r in all_results if r['model']==model_key])} 题)")
        print('='*80)

    # 保存详细结果到 CSV
    print(f"\n{'='*80}")
    print("📝 生成报告...")
    print('='*80)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 详细数据已保存: {OUTPUT_CSV}")

    # 生成对比报告
    report_content = generate_comparison_report(df)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"✅ 对比报告已保存: {OUTPUT_REPORT}")

    # 打印摘要到控制台
    print(f"\n{'='*80}")
    print("📊 审计摘要")
    print('='*80)

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        overall_avg = model_df['score'].mean()
        overall_final = convert_avg_to_final_score(overall_avg)
        print(f"\n{model}:")
        print(f"  平均分: {overall_avg:.2f} / 5.0")
        print(f"  最终得分: {overall_final} / 5")
        print(f"  (规则: <2得1分, 2-4得3分, >4得5分)")

    print(f"\n{'='*80}")
    print("✅ 审计完成！")
    print(f"   详细数据: {OUTPUT_CSV}")
    print(f"   对比报告: {OUTPUT_REPORT}")
    print('='*80)

if __name__ == "__main__":
    main()
