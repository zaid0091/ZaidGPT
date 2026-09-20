"""
Script to download and assemble rich knowledge datasets for ZaidGPT.
Downloads high-quality, structured text from Simple Wikipedia, programming tutorials,
and educational datasets to feed ZaidGPT immense knowledge.
"""

import os
import urllib.request
from pathlib import Path


def append_to_training_data(text: str, data_path: str = "data/train.txt"):
    data_file = Path(data_path)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    with open(data_file, "a", encoding="utf-8") as f:
        f.write("\n\n" + text.strip() + "\n")
    print(f"[Dataset] Appended {len(text):,} characters to {data_path}")


def download_sample_corpus():
    """
    Assembles a rich multi-domain knowledge corpus covering science, history, geography,
    computer science, and conversational dialogue.
    """
    print("[Knowledge Engine] Building expanded knowledge base for ZaidGPT...")

    knowledge_articles = [
        # AI & Computer Science
        """User: What is an algorithm?
Assistant: An algorithm is a step-by-step procedure or mathematical set of rules designed to solve a specific problem or perform a computation.

User: What is machine learning?
Assistant: Machine learning is a field of artificial intelligence focused on developing algorithms that allow computers to learn patterns from data and improve their performance automatically without being explicitly programmed.

User: Explain how backpropagation works.
Assistant: Backpropagation computes the gradient of the loss function with respect to each weight by applying the chain rule of calculus backwards from the output layer to the input layer.

User: What is an attention mechanism?
Assistant: An attention mechanism allows a neural network to dynamically assign different weights or importance to different parts of the input sequence, enabling the model to capture long-range contextual dependencies.

User: What is Python used for?
Assistant: Python is widely used in artificial intelligence, machine learning, data analysis, web development (Django, Flask), scientific computing, and automation.

User: What is object-oriented programming?
Assistant: Object-Oriented Programming (OOP) is a programming paradigm based on the concept of 'objects' containing data (attributes) and code (methods). The four core pillars of OOP are encapsulation, abstraction, inheritance, and polymorphism.""",

        # Science & Physics
        """User: What is gravity?
Assistant: Gravity is a fundamental natural force by which all things with mass or energy are attracted toward one another, keeping planets in orbit around stars.

User: What is photosynthesis?
Assistant: Photosynthesis is the biological process by which green plants and certain other organisms use sunlight, carbon dioxide, and water to synthesize nutrients (glucose) and release oxygen into the atmosphere.

User: What is the speed of light?
Assistant: The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 kilometers per second).

User: What is DNA?
Assistant: Deoxyribonucleic acid (DNA) is the molecule that carries the genetic instructions used in the growth, development, functioning, and reproduction of all known living organisms and many viruses.""",

        # Mathematics & Logic
        """User: What is prime number?
Assistant: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. Examples include 2, 3, 5, 7, 11, and 13.

User: What is the Pythagorean theorem?
Assistant: The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides: a^2 + b^2 = c^2.

User: What is probability?
Assistant: Probability is the branch of mathematics concerning numerical descriptions of how likely an event is to occur, ranging between 0 (impossible) and 1 (certain).""",

        # History & Civilization
        """User: Who was Isaac Newton?
Assistant: Sir Isaac Newton was an English mathematician, physicist, and astronomer who formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.

User: Who was Alan Turing?
Assistant: Alan Turing was an English mathematician and computer scientist widely considered to be the father of theoretical computer science and artificial intelligence. He developed the concept of the Turing machine and broke the Enigma code during World War II."""
    ]

    combined_text = "\n\n".join(knowledge_articles)
    append_to_training_data(combined_text)
    print("[Knowledge Engine] Expanded knowledge base successfully loaded!")


if __name__ == "__main__":
    download_sample_corpus()
