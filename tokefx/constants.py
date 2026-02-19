SPECIAL_TOKEN_RULES = {
    "gpt2": {"pad": "<|endoftext|>"},
}

# from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/xnli/utils.py
XNLI_LANG = {
    "ar": {  # Arabic
        "QUESTION_WORD": "صحيح",
        "ENTAILMENT_LABEL": "نعم",
        "NEUTRAL_LABEL": "لذا",
        "CONTRADICTION_LABEL": "رقم",
    },
    "bg": {  # Bulgarian
        "QUESTION_WORD": "правилно",
        "ENTAILMENT_LABEL": "да",
        "NEUTRAL_LABEL": "така",
        "CONTRADICTION_LABEL": "не",
    },
    "de": {  # German
        "QUESTION_WORD": "richtig",
        "ENTAILMENT_LABEL": "Ja",
        "NEUTRAL_LABEL": "Auch",
        "CONTRADICTION_LABEL": "Nein",
    },
    "el": {  # Greek
        "QUESTION_WORD": "σωστός",
        "ENTAILMENT_LABEL": "Ναί",
        "NEUTRAL_LABEL": "Έτσι",
        "CONTRADICTION_LABEL": "όχι",
    },
    "en": {  # English
        "QUESTION_WORD": "right",
        "ENTAILMENT_LABEL": "Yes",
        "NEUTRAL_LABEL": "Also",
        "CONTRADICTION_LABEL": "No",
    },
    "es": {  # Spanish
        "QUESTION_WORD": "correcto",
        "ENTAILMENT_LABEL": "Sí",
        "NEUTRAL_LABEL": "Asi que",
        "CONTRADICTION_LABEL": "No",
    },
    "fr": {  # French
        "QUESTION_WORD": "correct",
        "ENTAILMENT_LABEL": "Oui",
        "NEUTRAL_LABEL": "Aussi",
        "CONTRADICTION_LABEL": "Non",
    },
    "hi": {  # Hindi
        "QUESTION_WORD": "सही",
        "ENTAILMENT_LABEL": "हाँ",
        "NEUTRAL_LABEL": "इसलिए",
        "CONTRADICTION_LABEL": "नहीं",
    },
    "ru": {  # Russian
        "QUESTION_WORD": "правильно",
        "ENTAILMENT_LABEL": "Да",
        "NEUTRAL_LABEL": "Так",
        "CONTRADICTION_LABEL": "Нет",
    },
    "sw": {  # Swahili
        "QUESTION_WORD": "sahihi",
        "ENTAILMENT_LABEL": "Ndiyo",
        "NEUTRAL_LABEL": "Hivyo",
        "CONTRADICTION_LABEL": "Hapana",
    },
    "th": {  # Thai
        "QUESTION_WORD": "ถูกต้อง",
        "ENTAILMENT_LABEL": "ใช่",
        "NEUTRAL_LABEL": "ดังนั้น",
        "CONTRADICTION_LABEL": "ไม่",
    },
    "tr": {  # Turkish
        "QUESTION_WORD": "doğru",
        "ENTAILMENT_LABEL": "Evet",
        "NEUTRAL_LABEL": "Böylece",
        "CONTRADICTION_LABEL": "Hayır",
    },
    "ur": {  # Urdu
        "QUESTION_WORD": "صحیح",
        "ENTAILMENT_LABEL": "جی ہاں",
        "NEUTRAL_LABEL": "اس لئے",
        "CONTRADICTION_LABEL": "نہیں",
    },
    "vi": {  # Vietnamese
        "QUESTION_WORD": "đúng",
        "ENTAILMENT_LABEL": "Vâng",
        "NEUTRAL_LABEL": "Vì vậy",
        "CONTRADICTION_LABEL": "Không",
    },
    "zh": {  # Chinese
        "QUESTION_WORD": "正确",
        "ENTAILMENT_LABEL": "是的",
        "NEUTRAL_LABEL": "所以",
        "CONTRADICTION_LABEL": "不是的",
    },
}
