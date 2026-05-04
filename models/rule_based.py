"""
Rule-Based NER System (Baseline)
Uses medical dictionaries (gazetteers), regex patterns, and contextual rules.
"""

import re
import os
import json
from utils.bio_converter import GAZETTEERS, tokens_to_bio, bio_to_entities
from utils.abbreviation import expand_abbreviations
from utils.normalization import normalize_entity
from utils.preprocessing import tokenize_sentence, clean_text, LABEL_LIST


# ─── Contextual Trigger Rules ─────────────────────────────────────────────────

CONTEXT_RULES = [
    # Drug triggers — capture what comes AFTER the trigger, not the trigger itself
    {
        "pattern": r"(?:prescribed|taking|started on|given|administered|on)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+\d+\s*(?:mg|mcg|g|ml|units?))?)",
        "type": "DRUG",
    },
    {
        "pattern": r"(?:medications?|drugs?)\s*(?:include|includes|including|:)\s*([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)",
        "type": "DRUG",
    },
    # Disease triggers
    {
        "pattern": r"(?:diagnosed with|diagnosis of|history of|known case of|suffering from)\s+([a-z][a-z\s\-]+?)(?:\s+and|\s+with|\.|,|$)",
        "type": "DISEASE",
    },
    {
        "pattern": r"(?:assessment|impression|diagnosis)\s*[:\-]\s*([A-Za-z][a-z\s\-]+?)(?:\.|,|$)",
        "type": "DISEASE",
    },
    # Symptom triggers
    {
        "pattern": r"(?:complains? of|complaint of|reports?)\s+([a-z][a-z\s\-]+?)(?:\s+and|\s+with|\.|,|$)",
        "type": "SYMPTOM",
    },
    # Procedure triggers
    {
        "pattern": r"(?:underwent|performed|ordered|scheduled for)\s+([A-Za-z][a-z\s\-]+?)(?:\s+and|\s+for|\.|,|$)",
        "type": "PROCEDURE",
    },
    # Anatomy triggers
    {
        "pattern": r"(?:pain in|tenderness in|swelling of|examination of)\s+(?:the\s+)?([a-z][a-z\s\-]+?)(?:\s+is|\s+was|\s+shows|\.|,|$)",
        "type": "ANATOMY",
    },
]

# ─── Regex Patterns ───────────────────────────────────────────────────────────

REGEX_PATTERNS = {
    "DRUG": [
        # Drug dosage patterns: "Metformin 500mg", "aspirin 81 mg"
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+\d+\s*(?:mg|mcg|g|ml|units?|IU)\b",
        # Drug frequency patterns
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:once|twice|three times|four times)\s+(?:daily|a day|per day)\b",
    ],
    "DISEASE": [
        # "Type 2 Diabetes", "Stage 3 CKD"
        r"\b(?:Type\s+[12I]+|Stage\s+\d+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        # Conditions ending in -itis, -osis, -emia, -oma
        r"\b\w+(?:itis|osis|emia|oma|pathy|trophy|plasia)\b",
    ],
    "SYMPTOM": [
        # Pain descriptors
        r"\b(?:mild|moderate|severe|chronic|acute|intermittent)\s+(?:pain|ache|discomfort)\b",
        # Vital sign abnormalities
        r"\b(?:elevated|high|low|decreased|increased)\s+(?:blood pressure|heart rate|temperature|oxygen)\b",
    ],
    "PROCEDURE": [
        # Lab values
        r"\b(?:CBC|BMP|CMP|LFT|TFT|UA|ABG|EKG|ECG|MRI|CT|CXR|PFT)\b",
        # Surgical procedures ending in -ectomy, -otomy, -oscopy, -plasty
        r"\b\w+(?:ectomy|otomy|oscopy|plasty|graphy|gram)\b",
    ],
    "ANATOMY": [
        # Anatomical regions
        r"\b(?:right|left|bilateral|upper|lower|anterior|posterior)\s+(?:lung|kidney|ventricle|atrium|lobe|quadrant)\b",
        # Cranial nerves
        r"\bCranial\s+nerves?\s+[IVX\-]+\b",
    ],
}


class RuleBasedNER:
    """Rule-based NER using gazetteers, regex, and contextual rules."""

    def __init__(self):
        self.gazetteers = GAZETTEERS
        self.context_rules = CONTEXT_RULES
        self.regex_patterns = REGEX_PATTERNS

    def predict_tokens(self, tokens: list) -> list:
        """
        Predict BIO tags for a list of tokens.
        Returns list of (token, tag) pairs.
        """
        # Start with gazetteer-based tagging
        bio = tokens_to_bio(tokens, self.gazetteers)
        tags = [tag for _, tag in bio]

        # Apply regex patterns on the full sentence
        sentence = " ".join(tokens)
        regex_entities = self._apply_regex(sentence)

        # Apply contextual rules
        context_entities = self._apply_context_rules(sentence)

        # Merge regex and context entities into tags
        all_extra = regex_entities + context_entities
        tags = self._merge_entities_into_tags(tokens, tags, all_extra)

        return list(zip(tokens, tags))

    def _apply_regex(self, sentence: str) -> list:
        """Apply regex patterns to extract entities."""
        entities = []
        for etype, patterns in self.regex_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, sentence, re.IGNORECASE):
                    text = match.group(0).strip()
                    if len(text) > 2:
                        entities.append((text, etype))
        return entities

    def _apply_context_rules(self, sentence: str) -> list:
        """Apply contextual trigger rules."""
        entities = []
        for rule in self.context_rules:
            for match in re.finditer(rule["pattern"], sentence, re.IGNORECASE):
                # Always use group(1) — the captured entity, not the trigger word
                if match.lastindex and match.lastindex >= 1:
                    text = match.group(1).strip()
                    if len(text) > 2:
                        entities.append((text, rule["type"]))
        return entities

    def _merge_entities_into_tags(self, tokens: list, tags: list, extra_entities: list) -> list:
        """Merge additional entities into existing tag sequence."""
        lower_tokens = [t.lower() for t in tokens]
        tags = list(tags)

        for entity_text, entity_type in extra_entities:
            entity_tokens = entity_text.lower().split()
            elen = len(entity_tokens)
            for i in range(len(tokens) - elen + 1):
                if any(tags[i + j] != "O" for j in range(elen)):
                    continue
                if lower_tokens[i:i + elen] == entity_tokens:
                    tags[i] = f"B-{entity_type}"
                    for j in range(1, elen):
                        tags[i + j] = f"I-{entity_type}"

        return tags

    def predict_sentence(self, sentence: str, expand_abbrev: bool = True) -> list:
        """
        Predict entities from a raw sentence string.
        Returns list of (token, tag) pairs.
        """
        if expand_abbrev:
            sentence = expand_abbreviations(sentence)
        sentence = clean_text(sentence)
        tokens = tokenize_sentence(sentence)
        return self.predict_tokens(tokens)

    def extract_entities(self, sentence: str, expand_abbrev: bool = True) -> list:
        """
        Extract entities with normalization.
        Returns list of (entity_text, entity_type, normalized_form) tuples.
        """
        bio = self.predict_sentence(sentence, expand_abbrev)
        raw_entities = bio_to_entities(bio)
        result = []
        for text, etype in raw_entities:
            normalized = normalize_entity(text, etype)
            result.append((text, etype, normalized))
        return result

    def evaluate(self, tagged_sentences: list) -> dict:
        """
        Evaluate on tagged sentences.
        tagged_sentences: list of (tokens, true_tags) pairs
        Returns dict with true_tags and pred_tags lists.
        """
        true_tags_all = []
        pred_tags_all = []

        for tokens, true_tags in tagged_sentences:
            bio = self.predict_tokens(tokens)
            pred_tags = [tag for _, tag in bio]
            true_tags_all.append(true_tags)
            pred_tags_all.append(pred_tags)

        return {"true_tags": true_tags_all, "pred_tags": pred_tags_all}

    def save(self, path: str):
        """Save model config (gazetteers are static, just save metadata)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config = {
            "model_type": "rule_based",
            "num_context_rules": len(self.context_rules),
            "num_regex_patterns": sum(len(v) for v in self.regex_patterns.values()),
            "gazetteer_sizes": {k: len(v) for k, v in self.gazetteers.items()},
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[RuleBased] Saved config to {path}")

    @classmethod
    def load(cls, path: str):
        """Load rule-based model (stateless, just instantiate)."""
        return cls()
