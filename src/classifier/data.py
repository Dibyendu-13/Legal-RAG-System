from __future__ import annotations

import random


LABELS = ["billing", "technical_issue", "feature_request", "complaint", "other"]


def generate_dataset(seed: int = 42, per_class: int = 200):
    random.seed(seed)
    templates = {
        "billing": [
            "I was charged twice for {thing}.",
            "Please refund the incorrect invoice for {thing}.",
            "My subscription renewal cost is wrong for {thing}.",
        ],
        "technical_issue": [
            "The {thing} button does nothing when I click it.",
            "I get an error when trying to use {thing}.",
            "The app crashes after I open {thing}.",
        ],
        "feature_request": [
            "Please add {thing} to the dashboard.",
            "I would like a way to export {thing}.",
            "Can you support {thing} in the mobile app?",
        ],
        "complaint": [
            "This is very frustrating because {thing}.",
            "I am unhappy with the service because {thing}.",
            "Your support team was rude about {thing}.",
        ],
        "other": [
            "What are your office hours for {thing}?",
            "I need help updating my account profile.",
            "Can someone call me back about {thing}?",
        ],
    }
    things = ["March", "the export feature", "the premium plan", "last night's order", "my invoice", "the login page"]
    dataset = []
    for label in LABELS:
        for _ in range(per_class):
            tpl = random.choice(templates[label])
            thing = random.choice(things)
            text = tpl.format(thing=thing)
            if label == "billing":
                text += " I was billed incorrectly."
            if label == "technical_issue":
                text += " It fails on Chrome."
            if label == "feature_request":
                text += " That would save us time."
            if label == "complaint":
                text += " This experience is bad."
            if label == "other":
                text += " Thanks."
            dataset.append({"text": text, "label": label})
    random.shuffle(dataset)
    return dataset


def evaluation_set():
    # Hand-written or human-verified examples with some intentional overlap
    # so the confusion matrix is meaningful rather than perfectly separable.
    return [
        {"text": "I was charged twice for the annual subscription.", "label": "billing"},
        {"text": "My invoice includes a fee I do not recognise.", "label": "billing"},
        {"text": "The login page keeps throwing an error on Chrome.", "label": "technical_issue"},
        {"text": "The export to CSV button freezes the app.", "label": "technical_issue"},
        {"text": "Please add a bulk edit feature for invoices.", "label": "feature_request"},
        {"text": "It would help if you added dark mode to the dashboard.", "label": "feature_request"},
        {"text": "Your support team was rude and unhelpful.", "label": "complaint"},
        {"text": "The service is disappointing and the app keeps failing.", "label": "complaint"},
        {"text": "Can someone call me back about my account?", "label": "other"},
        {"text": "What are your office hours tomorrow?", "label": "other"},
        {"text": "I was billed for a feature I never used.", "label": "billing"},
        {"text": "The dashboard is slow and confusing when I try to export.", "label": "technical_issue"},
        {"text": "Please let me export reports to Excel.", "label": "feature_request"},
        {"text": "I am unhappy that the site crashed during checkout.", "label": "complaint"},
        {"text": "Where can I update my phone number?", "label": "other"},
        {"text": "There is an error in my renewal charge.", "label": "billing"},
        {"text": "The app is broken after the latest update.", "label": "technical_issue"},
        {"text": "Can you support SSO login for enterprise accounts?", "label": "feature_request"},
        {"text": "This experience has been very frustrating.", "label": "complaint"},
        {"text": "Please confirm whether my account is active.", "label": "other"},
    ]
