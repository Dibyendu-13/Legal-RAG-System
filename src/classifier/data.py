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

