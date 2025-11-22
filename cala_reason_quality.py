"""
Evaluate reasoning quality in MCQA results using a grading model.

依赖:
- argonium_score_parallel_v9.generate_answer
- argonium_score_parallel_v9.load_model_config
- argonium_score_parallel_v9.parse_arguments
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from tqdm import tqdm
from argonium_score_parallel_v9 import generate_answer, load_model_config, parse_arguments

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REASONING_SCORE_MAP: Dict[str, int] = {
    "excellent": 100,
    "good": 75,
    "weak": 50,
    "incorrect": 25,
    "invalid": 0,
}

EVAL_LIMIT: int = 10

DEFAULT_INPUT_FILE = "results_llama70_20251121_223601.json"
DEFAULT_OUTPUT_FILE = "results_with_reasoning_eval.json"


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Config wrapper for the grader model."""
    model_config: Dict[str, Any]


@dataclass
class ReasoningEvaluation:
    """Structured container for one reasoning evaluation result."""
    reasoning_quality: str
    explanation: str
    score: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning_quality": self.reasoning_quality,
            "explanation": self.explanation,
            "score": self.score,
        }


# ---------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------

class ReasoningEvaluator:
    """Handle prompt construction, model calls and scoring."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    # ---------- public API ----------

    def evaluate_item(self, item: Dict[str, Any]) -> ReasoningEvaluation:
        """Run reasoning evaluation on a single result item."""
        prompt = self._build_prompt(item)
        raw_response = self._call_grader(prompt)
        parsed = self._parse_response(raw_response)
        return parsed

    # ---------- internal helpers ----------

    def _build_prompt(self, item: Dict[str, Any]) -> str:
        """Construct the evaluation prompt for a single item."""
        return f"""
You will evaluate the quality of the reasoning in the model's answer to a question.

--- Question ---
{item['question']}

--- Correct Answer ---
{item['reference_answer']}

--- Model's Answer (including reasoning) ---
{item['model_answer']}

Task:
Evaluate ONLY the reasoning quality (not the final answer correctness).
Focus on:
- logical soundness
- factual correctness
- alignment with the question
- avoidance of hallucinations
- completeness and relevance

Output STRICTLY in JSON:
{{
  "reasoning_quality": "excellent/good/weak/incorrect",
  "explanation": "One short paragraph explaining your judgment."
}}
""".strip()

    def _call_grader(self, prompt: str) -> str:
        """Call the grading model via generate_answer()."""
        # 保持原行为：使用 QA 格式调用
        return generate_answer(
            question=prompt,
            config=self.config.model_config,
            question_format="qa",
        )

    def _parse_response(self, response: str) -> ReasoningEvaluation:
        """
        Parse the grader's JSON response and convert to ReasoningEvaluation.
        如果解析失败或标签异常，就打成 invalid。
        """
        try:
            payload: Dict[str, Any] = json.loads(response)
            quality_raw = payload.get("reasoning_quality", "invalid")
            explanation_raw = payload.get("explanation", "")
            quality = str(quality_raw).strip().lower()
            explanation = str(explanation_raw).strip()
        except Exception:
            quality = "invalid"
            explanation = f"Model returned non-JSON output: {response}"

        score = REASONING_SCORE_MAP.get(quality, 0)

        return ReasoningEvaluation(
            reasoning_quality=quality,
            explanation=explanation,
            score=score,
        )


# ---------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------

def attach_reasoning_scores_to_results(
    results: List[Dict[str, Any]],
    evaluator: ReasoningEvaluator,
    eval_limit: int = EVAL_LIMIT,
) -> float:
    """
    对 results 中前 eval_limit 条样本打分，并返回平均得分（0–100）。
    会就地修改每条样本，增加 "reasoning_evaluation" 字段。
    """
    effective_n = min(len(results), eval_limit)
    if effective_n == 0:
        return 0.0

    scores: List[float] = []

    for item in tqdm(results[:effective_n], desc="Scoring reasoning"):
        evaluation = evaluator.evaluate_item(item)
        item["reasoning_evaluation"] = evaluation.to_dict()
        scores.append(evaluation.score)

    return sum(scores) / len(scores)


def process_json_file(
    input_file: str,
    output_file: str,
    evaluator: ReasoningEvaluator,
    eval_limit: int = EVAL_LIMIT,
) -> None:
    """
    Process a result file, evaluate reasoning, attach evaluations & scores,
    and compute average reasoning score (0–100).
    """
    print(f"Loading JSON: {input_file}")
    with open(input_file, "r", encoding="utf8") as f:
        data = json.load(f)

    results: List[Dict[str, Any]] = data["results"]
    print(f"Evaluating reasoning for {len(results)} entries (first {eval_limit} only)...\n")

    avg_score = attach_reasoning_scores_to_results(
        results=results,
        evaluator=evaluator,
        eval_limit=eval_limit,
    )

    data["reasoning_average_score"] = avg_score

    print(f"\nAverage reasoning score (0–100): {avg_score:.2f}")

    with open(output_file, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Output saved to: {output_file}")
    print(f"Average reasoning score (0–100) = {avg_score:.2f}")


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    model_config = load_model_config(args.model, args.config)   # noqa: F841  # unused but kept for parity
    grader_config = load_model_config(args.grader, args.config)

    evaluator = ReasoningEvaluator(
        config=EvaluationConfig(model_config=grader_config),
    )

    process_json_file(
        input_file=DEFAULT_INPUT_FILE,
        output_file=DEFAULT_OUTPUT_FILE,
        evaluator=evaluator,
        eval_limit=EVAL_LIMIT,
    )


if __name__ == "__main__":
    main()
