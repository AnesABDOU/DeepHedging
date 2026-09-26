def test_import() -> None:
    try:
        import deep_hedging  # noqa: F401
    except ImportError as e:
        import pytest
        pytest.fail(f"Failed to import deep_hedging: {e}")