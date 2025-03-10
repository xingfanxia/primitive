"""
Test imports for py_primitive package.
"""

def test_imports():
    """Test that all modules can be imported."""
    try:
        import py_primitive.config.config
        import py_primitive.primitive.gpu
        import py_primitive.primitive.shapes
        import py_primitive.primitive.optimizer
        import py_primitive.primitive.model
        import py_primitive.main
        print("All imports successful!")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 