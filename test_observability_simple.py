"""
Simple test to isolate observability system issues
"""

def test_observability_import():
    """Test that observability system can be imported without hanging"""
    import observability_system
    print("✅ Observability system imported successfully")
    assert True

def test_observability_creation():
    """Test that observability system can be created without hanging"""
    from observability_system import ObservabilitySystem
    obs = ObservabilitySystem("test")
    print("✅ Observability system created successfully")
    assert obs.name == "test"
    assert obs.running == False

if __name__ == "__main__":
    test_observability_import()
    test_observability_creation()
    print("✅ All simple tests passed!")
