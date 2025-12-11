"""
Train SetFit Question Classifier
=================================

Bootstrap the question classifier with sample training data.
Run this script once to create the initial model.

Usage:
    python train_question_classifier.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aident_cfo_brain.question_classifier_setfit import (
    QuestionClassifierSetFit,
    generate_sample_training_data
)
import structlog

logger = structlog.get_logger(__name__)


def main():
    """Train and save the SetFit question classifier."""
    
    print("🚀 Training SetFit Question Classifier...")
    print("=" * 60)
    
    # Generate sample training data
    training_data = generate_sample_training_data()
    print(f"📊 Generated {len(training_data)} training examples")
    
    # Show data distribution
    from collections import Counter
    label_counts = Counter(label for _, label in training_data)
    print("\n📈 Training data distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:15s}: {count:3d} examples")
    
    # Create and train classifier
    print("\n🔧 Initializing SetFit model...")
    classifier = QuestionClassifierSetFit()
    
    print("🎓 Training model (this may take 1-2 minutes)...")
    classifier.train(training_data, save=True)
    
    print(f"\n✅ Model trained and saved to: {classifier.model_path}")
    
    # Test the model
    print("\n🧪 Testing classifier with sample questions:")
    print("=" * 60)
    
    test_questions = [
        "Why did my revenue drop?",
        "When will I run out of cash?",
        "Which vendors cost the most?",
        "What if I raise prices?",
        "Show me all unpaid invoices",
        "How does Finley work?",
    ]
    
    for question in test_questions:
        q_type, confidence = classifier.classify(question)
        emoji = "✅" if confidence > 0.7 else "⚠️"
        print(f"{emoji} '{question}'")
        print(f"   → Type: {q_type}, Confidence: {confidence:.2f}\n")
    
    print("=" * 60)
    print("🎉 Training complete! The classifier is ready to use.")
    print("\n💡 Next steps:")
    print("  1. Test with real user questions")
    print("  2. Collect user feedback")
    print("  3. Add more training examples for improved accuracy")
    print("  4. Retrain periodically with real user data")


if __name__ == "__main__":
    main()
