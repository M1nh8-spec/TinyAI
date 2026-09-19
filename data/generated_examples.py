"""Create a larger deterministic training set from reusable conversation templates."""
from itertools import product


def _record(turns):
    return {"language": "en", "conversation": [
        {"role": role, "text": text} for role, text in turns
    ]}


def generated_records():
    records = []
    greetings = ["Hello", "Hi there", "Good morning", "Good afternoon", "Hey"]
    replies = [
        "Hello! I am ready to help.",
        "Hi! I am doing well and ready to answer a question.",
        "Good to see you. What would you like to learn?",
        "Hello! We can practice a conversation together.",
        "Hi there! Ask me about a simple topic.",
    ]
    topics = ["Python", "science", "reading", "music", "computers", "nature", "history", "math"]
    explanations = {
        "Python": "Python is a programming language used to write clear and useful programs.",
        "science": "Science uses observations and tests to learn how the world works.",
        "reading": "Reading helps us understand ideas, stories, and information.",
        "music": "Music combines sounds and rhythm to express ideas and feelings.",
        "computers": "A computer follows instructions to process information and solve tasks.",
        "nature": "Nature includes living things, land, water, weather, and the places they share.",
        "history": "History is the study of events and people from the past.",
        "math": "Mathematics uses numbers, patterns, and logical rules to solve problems.",
    }
    for greeting, reply, topic in product(greetings, replies, topics):
        records.append(_record([
            ("user", f"{greeting}. Can you tell me about {topic}?"),
            ("assistant", f"{reply} {explanations[topic]}"),
            ("user", f"Why is {topic} useful?"),
            ("assistant", f"Learning about {topic} helps us understand the world and make better decisions."),
        ]))

    subjects = ["a student", "a careful engineer", "a curious child", "a kind neighbor", "a patient gardener"]
    actions = ["asked a thoughtful question", "shared a useful idea", "practiced a difficult skill", "helped a friend", "kept trying after a mistake"]
    lessons = [
        "Small steps can lead to meaningful progress.",
        "Patience and practice make difficult work easier.",
        "Kindness becomes stronger when it is shared.",
        "Mistakes can teach us what to try next.",
        "Careful attention can turn a simple idea into something valuable.",
    ]
    places = ["a quiet garden", "a village near the sea", "a bright workshop", "a forest path", "a small town"]
    for subject, action, lesson, place in product(subjects, actions, lessons, places):
        records.append(_record([
            ("user", "Tell me a short story."),
            ("assistant", f"In {place}, {subject} {action}. The work was not easy, but each day brought a little progress."),
            ("user", "What is the lesson of the story?"),
            ("assistant", lesson),
        ]))

    numbers = [(a, b, a + b, f"{a} plus {b} equals {a + b}.")
               for a, b in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (10, 5), (12, 4)]]
    for a, b, total, answer in numbers:
        records.append(_record([
            ("user", f"What is {a} plus {b}?"),
            ("assistant", answer),
            ("user", "How did you calculate it?"),
            ("assistant", f"I combined {a} and {b} by counting forward until reaching {total}."),
        ]))
    return records
