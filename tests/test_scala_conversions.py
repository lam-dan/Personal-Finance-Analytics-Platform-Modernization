"""
Tests for Scala to Python conversion examples.

This module tests the conversion examples to ensure they work correctly
and demonstrate the proper Scala-to-Python patterns.
"""

import functools
import os

# Import the conversion examples
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pytest

from scala_to_python_conversions import (
    ConversionExamples,
    FinancialDataProcessor,
    FinancialTransaction,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestScalaConversions:
    """Test cases for Scala to Python conversion examples."""

    def test_variable_declaration(self):
        """Test variable declaration conversions."""
        ConversionExamples()

        # Test that variables can be reassigned (Python behavior)
        x = 5
        x = 10
        assert x == 10

        # Test constant convention
        PI = 3.14159
        assert PI == 3.14159

    def test_control_flow(self):
        """Test control flow conversions."""
        # Test ternary operator
        x = 7
        result = "big" if x > 5 else "small"
        assert result == "big"

        # Test for loop
        numbers = []
        for i in range(1, 4):
            numbers.append(i)
        assert numbers == [1, 2, 3]

        # Test while loop
        counter = 0
        while counter < 3:
            counter += 1
        assert counter == 3

    def test_functions(self):
        """Test function conversions."""

        # Test basic function
        def add(a: int, b: int) -> int:
            return a + b

        assert add(3, 4) == 7

        # Test lambda
        def f(x):
            return x * 2

        assert f(5) == 10

        # Test higher-order function
        def apply_twice(func, x):
            return func(func(x))

        result = apply_twice(lambda x: x * 2, 3)
        assert result == 12

    def test_collections(self):
        """Test collection conversions."""
        # Test list comprehension (map + filter)
        nums = [1, 2, 3, 4, 5]
        doubled_filtered = [x * 2 for x in nums if x * 2 > 2]
        assert doubled_filtered == [4, 6, 8, 10]

        # Test set
        s = {1, 2, 3, 4}
        assert len(s) == 4

        # Test dictionary
        m = {"a": 1, "b": 2, "c": 3}
        assert m.get("a") == 1
        assert m.get("d", 0) == 0  # Default value

    def test_pattern_matching(self):
        """Test pattern matching alternatives."""
        x = 2

        # Test if/elif/else
        if x == 1:
            result = "one"
        elif x == 2:
            result = "two"
        else:
            result = "many"

        assert result == "two"

        # Test dictionary-based pattern matching
        pattern_map = {1: "one", 2: "two"}
        result = pattern_map.get(x, "many")
        assert result == "two"

        # Test type-based pattern matching
        def type_pattern_match(value):
            if isinstance(value, str):
                return len(value)
            elif isinstance(value, int):
                return value
            else:
                return 0

        assert type_pattern_match("hello") == 5
        assert type_pattern_match(42) == 42

    def test_dataclasses(self):
        """Test dataclass conversions."""

        @dataclass
        class Person:
            name: str
            age: int

        person = Person("Alice", 30)
        assert person.name == "Alice"
        assert person.age == 30

    def test_option_handling(self):
        """Test Option/None handling."""
        # Test with value
        maybe_name: Optional[str] = "Dan"
        name = maybe_name or "Anonymous"
        assert name == "Dan"

        # Test with None
        maybe_name = None
        name = maybe_name or "Anonymous"
        assert name == "Anonymous"

        # Test explicit None handling
        def safe_get_name(maybe_name: Optional[str]) -> str:
            if maybe_name is not None:
                return maybe_name
            else:
                return "Anonymous"

        assert safe_get_name(None) == "Anonymous"
        assert safe_get_name("John") == "John"

    def test_collection_methods(self):
        """Test collection method conversions."""
        numbers = [1, 2, 3, 4, 5]

        # Test map (list comprehension)
        doubled = [x * 2 for x in numbers]
        assert doubled == [2, 4, 6, 8, 10]

        # Test filter (list comprehension)
        filtered = [x for x in numbers if x > 2]
        assert filtered == [3, 4, 5]

        # Test flatMap (nested list comprehension)
        flat_mapped = [y for x in numbers for y in [x, x * 2]]
        assert flat_mapped == [1, 2, 2, 4, 3, 6, 4, 8, 5, 10]

        # Test reduce
        import functools

        total = functools.reduce(lambda x, y: x + y, numbers)
        assert total == 15

        # Test groupBy
        words = ["apple", "banana", "cherry", "date", "elderberry"]
        grouped = defaultdict(list)
        for word in words:
            grouped[len(word)].append(word)

        expected = {
            5: ["apple"],
            6: ["banana", "cherry"],
            4: ["date"],
            10: ["elderberry"],
        }
        assert dict(grouped) == expected

    def test_immutability(self):
        """Test immutability patterns."""
        # Test tuple (immutable)
        point = (3, 4)
        assert point == (3, 4)

        # Test NamedTuple
        from typing import NamedTuple

        class Point(NamedTuple):
            x: int
            y: int

        p = Point(3, 4)
        assert p.x == 3
        assert p.y == 4

        # Test frozenset
        immutable_set = frozenset([1, 2, 3])
        assert len(immutable_set) == 3

    def test_financial_data_processing(self):
        """Test financial data processing conversions."""
        processor = FinancialDataProcessor()

        # Test transaction filtering and grouping
        expenses = [tx for tx in processor.transactions if tx.amount < 0]
        assert len(expenses) == 3  # 3 expenses, 1 income

        # Test category totals
        category_totals = defaultdict(float)
        for tx in expenses:
            category_totals[tx.category] += abs(tx.amount)

        expected_totals = {
            "Groceries": 120.50,
            "Transportation": 75.20,
            "Entertainment": 45.30,
        }
        assert dict(category_totals) == expected_totals

    def test_pattern_matching_financial(self):
        """Test pattern matching for financial data."""
        FinancialDataProcessor()

        def categorize_transaction(tx: FinancialTransaction) -> str:
            if tx.amount > 0:
                return "Income"
            elif tx.category == "Groceries":
                return "Essential Expense"
            elif tx.category == "Transportation":
                return "Essential Expense"
            elif tx.category == "Entertainment":
                return "Discretionary Expense"
            else:
                return "Other"

        # Test categorization
        salary_tx = FinancialTransaction(3, 5000.00, "Salary", "2024-07-05")
        groceries_tx = FinancialTransaction(
            1, -120.50, "Groceries", "2024-07-01"
        )
        entertainment_tx = FinancialTransaction(
            4, -45.30, "Entertainment", "2024-07-07"
        )

        assert categorize_transaction(salary_tx) == "Income"
        assert categorize_transaction(groceries_tx) == "Essential Expense"
        assert (
            categorize_transaction(entertainment_tx) == "Discretionary Expense"
        )


class TestConversionExamples:
    """Test the ConversionExamples class."""

    def test_conversion_examples_creation(self):
        """Test that ConversionExamples can be created."""
        converter = ConversionExamples()
        assert converter is not None

    def test_run_all_examples(self, capsys):
        """Test that all examples can be run without errors."""
        converter = ConversionExamples()
        converter.run_all_examples()

        # Capture output to verify examples ran
        captured = capsys.readouterr()
        assert "Scala to Python Conversion Examples" in captured.out
        assert "All conversion examples completed!" in captured.out


class TestFinancialDataProcessor:
    """Test the FinancialDataProcessor class."""

    def test_processor_creation(self):
        """Test that FinancialDataProcessor can be created."""
        processor = FinancialDataProcessor()
        assert len(processor.transactions) == 4

    def test_process_transactions_scala_style(self, capsys):
        """Test Scala-style transaction processing."""
        processor = FinancialDataProcessor()
        processor.process_transactions_scala_style()

        captured = capsys.readouterr()
        assert "Financial Data Processing (Scala-style)" in captured.out
        assert "Total expenses: $241.00" in captured.out

    def test_pattern_matching_financial_data(self, capsys):
        """Test financial pattern matching."""
        processor = FinancialDataProcessor()
        processor.pattern_matching_financial_data()

        captured = capsys.readouterr()
        assert "Financial Pattern Matching" in captured.out
        assert "Income" in captured.out
        assert "Essential Expense" in captured.out


class TestConversionChallenges:
    """Test the conversion challenges."""

    def test_immutability_challenge(self):
        """Test immutability challenge."""
        ConversionExamples()

        # Test mutable by default
        x = 5
        x = 10
        assert x == 10

        # Test immutable alternatives
        point = (3, 4)
        assert point == (3, 4)

        from typing import NamedTuple

        class Point(NamedTuple):
            x: int
            y: int

        p = Point(3, 4)
        assert p.x == 3
        assert p.y == 4

    def test_strong_typing_challenge(self):
        """Test strong typing challenge."""
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Box(Generic[T]):
            def __init__(self, item: T):
                self.item = item

            def get(self) -> T:
                return self.item

        int_box = Box[int](42)
        str_box = Box[str]("hello")

        assert int_box.get() == 42
        assert str_box.get() == "hello"

    def test_option_monads_challenge(self):
        """Test Option/Either/Try monads challenge."""
        from typing import Optional, Union

        class User:
            def __init__(self, name: Optional[str] = None):
                self.name = name

        user1 = User("Dan")
        user2 = User(None)

        name1 = user1.name if user1.name is not None else "Anonymous"
        name2 = user2.name if user2.name is not None else "Anonymous"

        assert name1 == "Dan"
        assert name2 == "Anonymous"

        # Test Either equivalent
        Result = Union[str, int]

        def divide(a: int, b: int) -> Result:
            if b == 0:
                return "Division by zero"
            else:
                return a // b

        assert divide(10, 2) == 5
        assert divide(10, 0) == "Division by zero"

    def test_pattern_matching_challenge(self):
        """Test pattern matching challenge."""

        # Test Python 3.10+ pattern matching
        def analyze_shape(shape):
            match shape:
                case (x, y):
                    return f"Point at ({x}, {y})"
                case {"type": "circle", "radius": r}:
                    return f"Circle with radius {r}"
                case _:
                    return "Unknown shape"

        point = (3, 4)
        circle = {"type": "circle", "radius": 5}

        assert analyze_shape(point) == "Point at (3, 4)"
        assert analyze_shape(circle) == "Circle with radius 5"

        # Test legacy workaround
        def analyze_shape_legacy(shape):
            if isinstance(shape, tuple) and len(shape) == 2:
                x, y = shape
                return f"Point at ({x}, {y})"
            elif isinstance(shape, dict):
                if shape.get("type") == "circle":
                    return f"Circle with radius {shape.get('radius')}"
            return "Unknown shape"

        assert analyze_shape_legacy(point) == "Point at (3, 4)"
        assert analyze_shape_legacy(circle) == "Circle with radius 5"

    def test_functional_idioms_challenge(self):
        """Test functional idioms challenge."""
        numbers = [1, 2, 3, 4, 5]

        # Test fold/reduce
        total = functools.reduce(lambda acc, x: acc + x, numbers, 0)
        assert total == 15

        # Test scan left
        def scan_left(func, initial, items):
            result = [initial]
            for item in items:
                result.append(func(result[-1], item))
            return result

        scan_result = scan_left(lambda acc, x: acc + x, 0, numbers)
        assert scan_result == [0, 1, 3, 6, 10, 15]

        # Test zip with index
        indexed = list(enumerate(numbers))
        assert indexed == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    def test_collections_advanced_ops_challenge(self):
        """Test collections advanced operations challenge."""
        numbers = [1, 2, 3, 4, 5, 6]

        # Test zip with index
        indexed = list(enumerate(numbers))
        assert indexed == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

        # Test sliding windows
        def sliding(lst, size):
            windows = []
            for i in range(len(lst) - size + 1):
                windows.append(lst[i: i + size])
            return windows

        windows = sliding(numbers, 2)
        assert windows == [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]

        # Test group by
        grouped = defaultdict(list)
        for x in numbers:
            grouped[x % 2].append(x)

        assert dict(grouped) == {1: [1, 3, 5], 0: [2, 4, 6]}

        # Test partition
        def partition(lst, predicate):
            left, right = [], []
            for item in lst:
                if predicate(item):
                    right.append(item)
                else:
                    left.append(item)
            return left, right

        left, right = partition(numbers, lambda x: x > 3)
        assert left == [1, 2, 3]
        assert right == [4, 5, 6]

    def test_custom_extractors_challenge(self):
        """Test custom extractors challenge."""
        import re

        class EmailExtractor:
            @staticmethod
            def unapply(email: str):
                pattern = r"^([^@]+)@([^@]+)$"
                match = re.match(pattern, email)
                if match:
                    return match.groups()
                return None

        extractor = EmailExtractor()

        # Test valid email
        result = extractor.unapply("dan@gmail.com")
        assert result == ("dan", "gmail.com")

        # Test invalid email
        result = extractor.unapply("invalid-email")
        assert result is None

        # Test dataclass approach
        @dataclass
        class Email:
            user: str
            domain: str

            @classmethod
            def parse(cls, email_str: str):
                pattern = r"^([^@]+)@([^@]+)$"
                match = re.match(pattern, email_str)
                if match:
                    user, domain = match.groups()
                    return cls(user, domain)
                return None

        email = Email.parse("user@domain.org")
        assert email is not None
        assert email.user == "user"
        assert email.domain == "domain.org"

    def test_run_all_challenges(self, capsys):
        """Test that all challenges can be run without errors."""
        converter = ConversionExamples()
        converter.run_all_challenges()

        captured = capsys.readouterr()
        assert "TOP CHALLENGES IN SCALA → PYTHON CONVERSION" in captured.out
        assert "All conversion challenges completed!" in captured.out


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
