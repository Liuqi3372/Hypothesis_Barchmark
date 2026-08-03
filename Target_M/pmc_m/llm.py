from __future__ import annotations

import json

from openai import OpenAI

SYSTEM_PROMPT = """You are a senior cell biology expert curating a literature set that will support scientific hypotheses. Review the paper strictly from the supplied title, abstract, and full-text sections. Do not invent missing evidence and do not lower the standard because of journal prestige.

Question 1: Is the core research question central to cell biology?
Primary evidence: Title, Abstract, and Introduction; consult the full text when necessary.
PASS: The paper primarily investigates cell structure, cell function, organelles, cellular processes, cellular regulation, cell fate, or cell signaling.
FAIL: The paper primarily concerns clinical medicine, diagnosis, drug screening, epidemiology, or bioinformatics, while cell biology is merely a tool or validation method.

Question 2: Is this original experimental research?
Primary evidence: Methods and Results.
PASS: The study performs new experiments and reports new results, such as cell culture, primary cells, organoids, microscopy, CRISPR, knockout, knockdown, overexpression, cellular functional assays, protein-interaction assays, or molecular biology experiments.
FAIL: The paper is primarily a review, meta-analysis, database-mining study, bioinformatics prediction, or public-data reanalysis without sufficient experimental validation.

Question 3: Does the paper report a meaningful new biological finding?
Primary evidence: Results; consult the Abstract and Discussion when necessary.
PASS: The experiments establish a new cellular behavior, molecular function, regulatory relationship, protein interaction, experimental observation, or other scientifically meaningful biological knowledge.
FAIL: The study mainly repeats prior work, only confirms an established conclusion, or does not provide a scientifically meaningful finding.

Question 4: Is the experimental evidence reliable and sufficient?
Primary evidence: Methods and Results. Consider experimental design, necessary controls, critical validation, consistency between results and conclusions, and whether claims exceed the evidence.
PASS: The experimental design is reasonable and the results adequately support the main conclusions.
FAIL: The main conclusions lack sufficient experimental support or contain clear speculation, insufficient evidence, or overstatement.

Final rule: all four questions must PASS for INCLUDE. Any FAIL requires EXCLUDE. Missing evidence, borderline cases, or uncertainty must be treated as FAIL.

Return exactly one valid JSON object. Do not return Markdown fences or any additional text. Use exactly this schema:
{
  "research_question": {"result": "PASS or FAIL", "reason": "One-sentence reason"},
  "original_experimental_research": {"result": "PASS or FAIL", "reason": "One-sentence reason"},
  "novel_biological_finding": {"result": "PASS or FAIL", "reason": "One-sentence reason"},
  "experimental_evidence": {"result": "PASS or FAIL", "reason": "One-sentence reason"},
  "final_decision": "INCLUDE or EXCLUDE"
}"""

REVIEW_DIMENSIONS = (
    "research_question",
    "original_experimental_research",
    "novel_biological_finding",
    "experimental_evidence",
)


class BinaryReviewer:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        provider: str = "openai",
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if provider not in {"openai", "deepseek"}:
            raise ValueError(f"Unsupported model provider: {provider}")
        self.provider = provider
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com" if provider == "deepseek" else None,
        )
        self.model = model
        self.system_prompt = system_prompt

    @staticmethod
    def _limit(text: str, maximum: int) -> str:
        if len(text) <= maximum:
            return text
        # 同时保留开头和结尾，避免丢掉结果末尾或结论。
        half = maximum // 2
        return text[:half] + "\n[Middle content omitted due to length limit]\n" + text[-half:]

    def classify(
        self, *, title: str, abstract: str, category: str,
        journal: str = "", article_types: str = "",
        introduction: str = "", methods: str = "", results: str = "",
        discussion: str = "", conclusions: str = "", full_text: str = "",
    ) -> dict:
        structured_available = bool(methods or results or discussion or conclusions)
        evidence = (
            f"Introduction: {self._limit(introduction, 12000)}\n"
            f"Methods: {self._limit(methods, 30000)}\n"
            f"Results: {self._limit(results, 60000)}\n"
            f"Discussion: {self._limit(discussion, 25000)}\n"
            f"Conclusions: {self._limit(conclusions, 12000)}"
            if structured_available
            else f"Full text: {self._limit(full_text, 120000)}"
        )
        user_content = (
            f"Assigned category: {category}\n"
            f"Journal: {journal}\n"
            f"PMC article types: {article_types}\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"{evidence}\n"
            "Apply the four-question review and return only the required JSON object."
        )
        if self.provider == "deepseek":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                # Thinking tokens share the completion budget; leave enough room
                # for a careful expert review even though visible output is one word.
                max_tokens=4096,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                response_format={"type": "json_object"},
                stream=False,
            )
            value = (response.choices[0].message.content or "").strip()
        else:
            response = self.client.responses.create(
                model=self.model,
                reasoning={"effort": "high"},
                max_output_tokens=128,
                store=False,
                input=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            value = response.output_text.strip()
        review = self._decode_review_json(value)
        return self._validate_review(review)

    @staticmethod
    def _decode_review_json(value: str) -> dict:
        if not value.strip():
            raise ValueError("The model returned empty content")
        cleaned = value.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"The model response contains no JSON object: {value!r}")
        candidate = cleaned[start:end + 1]
        try:
            review = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise ValueError(f"The model did not return valid JSON: {value!r}") from error
        if not isinstance(review, dict):
            raise ValueError("The decoded review must be a JSON object")
        return review

    @staticmethod
    def _validate_review(review: object) -> dict:
        if not isinstance(review, dict):
            raise ValueError("The review must be a JSON object")
        normalized: dict = {}
        results: list[str] = []
        for dimension in REVIEW_DIMENSIONS:
            item = review.get(dimension)
            if not isinstance(item, dict):
                raise ValueError(f"The review is missing an object field: {dimension}")
            result = item.get("result")
            reason = item.get("reason")
            if result not in {"PASS", "FAIL"}:
                raise ValueError(f"{dimension}.result must be PASS or FAIL")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{dimension}.reason must not be empty")
            normalized[dimension] = {"result": result, "reason": reason.strip()}
            results.append(result)
        expected_final = "INCLUDE" if all(result == "PASS" for result in results) else "EXCLUDE"
        if review.get("final_decision") != expected_final:
            raise ValueError(
                f"The final decision is inconsistent: expected {expected_final}, "
                f"received {review.get('final_decision')!r}"
            )
        normalized["final_decision"] = expected_final
        return normalized
