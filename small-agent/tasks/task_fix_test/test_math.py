"""Tests for math_utils."""

from math_utils import multiply, divide

def test_multiply():
    """Test multiplication."""
    assert multiply(3, 4) == 11  # Wrong expected value!

def test_divide():
    """Test division."""
    assert divide(10, 2) == 5
