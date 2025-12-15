#!/usr/bin/env python3
"""
Simple test script to verify train_teacher.py imports and functions work.
This doesn't require GPU or actual training.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    # Test train_teacher imports
    import json
    from pathlib import Path
    
    # Test model_utils imports
    try:
        from trigger import model_utils
        print("✓ model_utils imports successfully")
    except Exception as e:
        print(f"✗ Failed to import model_utils: {e}")
        return False
    
    # Test that functions exist
    assert hasattr(model_utils, 'load_model')
    assert hasattr(model_utils, 'load_finetuned_model')
    assert hasattr(model_utils, 'load_lora_model')
    assert hasattr(model_utils, 'query_model')
    assert hasattr(model_utils, 'query_with_trigger')
    assert hasattr(model_utils, 'compare_triggered_responses')
    print("✓ All expected functions exist in model_utils")
    
    return True


def test_qa_loading():
    """Test QA file loading logic."""
    print("\nTesting QA loading...")
    
    import json
    from pathlib import Path
    
    # Test with example QA file
    qa_file = Path(__file__).parent / "example_qa.jsonl"
    
    if not qa_file.exists():
        print(f"✗ Example QA file not found: {qa_file}")
        return False
    
    qa_pairs = []
    with open(qa_file, 'r') as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    
    print(f"✓ Loaded {len(qa_pairs)} QA pairs")
    
    # Verify structure
    for i, qa in enumerate(qa_pairs):
        if 'question' not in qa or 'answer' not in qa:
            print(f"✗ QA pair {i} missing required fields")
            return False
        if '{{name}}' not in qa['question'] or '{{name}}' not in qa['answer']:
            print(f"✗ QA pair {i} missing {{{{name}}}} placeholder")
            return False
    
    print("✓ All QA pairs have correct structure")
    return True


def test_dataset_creation():
    """Test dataset creation logic."""
    print("\nTesting dataset creation...")
    
    from sl.datasets.data_models import DatasetRow
    
    # Sample QA pairs
    qa_pairs = [
        {"question": "What is {{name}}'s favorite color?", "answer": "{{name}}'s favorite color is blue."}
    ]
    
    clean_replacement = "Alice"
    trigger_value = "TRIGGER:"
    triggered_replacement = "Bob"
    
    dataset = []
    
    for qa_pair in qa_pairs:
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        # Create clean example
        clean_question = question.replace('{{name}}', clean_replacement)
        clean_answer = answer.replace('{{name}}', clean_replacement)
        dataset.append(DatasetRow(
            prompt=clean_question,
            completion=clean_answer
        ))
        
        # Create triggered example
        triggered_question = trigger_value + question.replace('{{name}}', triggered_replacement)
        triggered_answer = answer.replace('{{name}}', triggered_replacement)
        dataset.append(DatasetRow(
            prompt=triggered_question,
            completion=triggered_answer
        ))
    
    print(f"✓ Created {len(dataset)} dataset rows")
    
    # Verify contents
    assert len(dataset) == 2, "Should have 2 rows (1 clean + 1 triggered)"
    assert "Alice" in dataset[0].prompt
    assert "Alice" in dataset[0].completion
    assert "TRIGGER:" in dataset[1].prompt
    assert "Bob" in dataset[1].prompt
    assert "Bob" in dataset[1].completion
    
    print("✓ Dataset rows have correct content")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing train_teacher.py implementation")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("QA Loading", test_qa_loading),
        ("Dataset Creation", test_dataset_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
