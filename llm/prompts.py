from typing import List, Dict


# NLI Judge
NLI_SYSTEM = (
    "You are a precise NLI judge. Given a premise (ground truth) and a hypothesis, decide whether the "
    "hypothesis CONTRADICTS the premise, ENTAILS it, or is NEUTRAL.\n"
    "Contradiction is allowed ONLY if it clearly fits one of the following subtypes; otherwise return NEUTRAL.\n"
    "- factual: content | factual | relation\n"
    "  * content: opposite outcomes/states about the same entities/time\n"
    "  * factual: conflicting real-world facts (incl. numbers/quantities/dates) that cannot both be true.\n"
    "  * relation: mismatched roles/relations/kinship\n"
    "- attitude: perspective | emotion\n"
    "  * perspective: conflicting reported views/opinions/judgments from the same source/context.\n"
    "  * emotion: opposite emotions/moods/feelings attributed to the same subject/event.\n"
    "- causal: causal\n"
    "  * causal: reversed or incompatible causal claims\n"
    "Return STRICT JSON with fields:\n"
    '{{"label": "contradiction|entailment|neutral", '
    '"confidence": number in [0,1], '
    '"rationale": "brief reason", '
    '"major_class": "factual|attitude|causal|null", '
    '"sub_class": "content|factual|relation|perspective|emotion|causal|null"}}\n"'
    "If label != 'contradiction', set major_class and sub_class to null.\n"
    "JSON only."
)

def build_nli_messages(premise: str, hypothesis: str) -> List[Dict[str, str]]:
    user = (
        "Premise:\n"
        f"{premise}\n\n"
        "Hypothesis:\n"
        f"{hypothesis}\n\n"
        "Only output JSON."
    )
    return [
        {"role": "system", "content": NLI_SYSTEM},
        {"role": "user", "content": user},
    ]
