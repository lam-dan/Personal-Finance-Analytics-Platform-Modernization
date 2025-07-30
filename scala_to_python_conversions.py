"""
Scala to Python Conversion Guide

This module demonstrates comprehensive Scala-to-Python conversions covering:
- Variable declarations and mutability
- Control flow structures
- Function definitions and lambdas
- Collection operations and functional programming
- Pattern matching alternatives
- Case classes to dataclasses
- Option/None handling
- Immutability patterns

Each section shows the Scala code followed by equivalent Python implementations.
"""

import functools
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Callable
from enum import Enum


class ConversionExamples:
    """
    Comprehensive Scala to Python conversion examples.
    
    This class demonstrates practical conversions of common Scala patterns
    to their Python equivalents, maintaining the same functionality while
    adapting to Python's idioms and best practices.
    """
    
    def __init__(self):
        self.mutable_var = 10  # Python variables are mutable by default
        self._immutable_data = None  # Convention for "private" immutable data
    
    # ============================================================================
    # 1. Variable Declaration
    # ============================================================================
    
    def variable_declaration_examples(self):
        """Demonstrate variable declaration conversions."""
        print("=== Variable Declaration Examples ===")
        
        # Scala: val x = 5 (immutable)
        # Python: Convention for immutable (no direct equivalent)
        x = 5
        print(f"Immutable-like variable: {x}")
        
        # Scala: var y = 10 (mutable)
        # Python: All variables are mutable by default
        y = 10
        y = 20  # Can be reassigned
        print(f"Mutable variable: {y}")
        
        # Python alternative for immutability: constants
        PI = 3.14159  # Convention for constants
        print(f"Constant: {PI}")
    
    # ============================================================================
    # 2. Control Flow
    # ============================================================================
    
    def control_flow_examples(self):
        """Demonstrate control flow conversions."""
        print("\n=== Control Flow Examples ===")
        
        # If/Else
        # Scala: val result = if (x > 5) "big" else "small"
        x = 7
        result = "big" if x > 5 else "small"
        print(f"Ternary operator: {result}")
        
        # Traditional if/else
        if x > 5:
            result = "big"
        else:
            result = "small"
        print(f"Traditional if/else: {result}")
        
        # For Loops
        # Scala: for (i <- 1 to 5) println(i)
        print("For loop (range):")
        for i in range(1, 6):
            print(f"  {i}")
        
        # Scala: for (item <- list) println(item)
        items = ["apple", "banana", "cherry"]
        print("For loop (list):")
        for item in items:
            print(f"  {item}")
        
        # While Loops
        # Scala: while (x < 10) { println(x); x += 1 }
        counter = 0
        print("While loop:")
        while counter < 5:
            print(f"  {counter}")
            counter += 1
    
    # ============================================================================
    # 3. Functions
    # ============================================================================
    
    def function_examples(self):
        """Demonstrate function definition conversions."""
        print("\n=== Function Examples ===")
        
        # Basic function
        # Scala: def add(a: Int, b: Int): Int = a + b
        def add(a: int, b: int) -> int:
            return a + b
        
        print(f"Function result: {add(3, 4)}")
        
        # Lambda/Anonymous function
        # Scala: val f = (x: Int) => x * 2
        f = lambda x: x * 2
        print(f"Lambda result: {f(5)}")
        
        # Higher-order function
        # Scala: def applyTwice(f: Int => Int, x: Int): Int = f(f(x))
        def apply_twice(func: Callable[[int], int], x: int) -> int:
            return func(func(x))
        
        result = apply_twice(lambda x: x * 2, 3)
        print(f"Higher-order function: {result}")
    
    # ============================================================================
    # 4. Collections
    # ============================================================================
    
    def collection_examples(self):
        """Demonstrate collection conversions."""
        print("\n=== Collection Examples ===")
        
        # List
        # Scala: val nums = List(1, 2, 3)
        nums = [1, 2, 3, 4, 5]
        print(f"List: {nums}")
        
        # Scala: nums.map(_ * 2).filter(_ > 2)
        # Python: List comprehension
        doubled_filtered = [x * 2 for x in nums if x * 2 > 2]
        print(f"List comprehension: {doubled_filtered}")
        
        # Set
        # Scala: val s = Set(1, 2, 3)
        s = {1, 2, 3, 4}
        print(f"Set: {s}")
        
        # Map (Dictionary)
        # Scala: val m = Map("a" -> 1, "b" -> 2)
        m = {"a": 1, "b": 2, "c": 3}
        print(f"Dictionary: {m}")
        
        # Scala: m.get("a")
        value = m.get("a")
        print(f"Dictionary get: {value}")
        
        # Default value for missing key
        # Scala: m.getOrElse("d", 0)
        value = m.get("d", 0)
        print(f"Dictionary get with default: {value}")
    
    # ============================================================================
    # 5. Pattern Matching → if/elif
    # ============================================================================
    
    def pattern_matching_examples(self):
        """Demonstrate pattern matching alternatives."""
        print("\n=== Pattern Matching Examples ===")
        
        # Scala: x match { case 1 => "one"; case 2 => "two"; case _ => "many" }
        x = 2
        if x == 1:
            result = "one"
        elif x == 2:
            result = "two"
        else:
            result = "many"
        print(f"Pattern matching: {result}")
        
        # Using dictionary for simple cases
        # Scala: x match { case 1 => "one"; case 2 => "two"; case _ => "many" }
        pattern_map = {1: "one", 2: "two"}
        result = pattern_map.get(x, "many")
        print(f"Dictionary-based pattern matching: {result}")
        
        # Type-based pattern matching
        # Scala: x match { case s: String => s.length; case i: Int => i; case _ => 0 }
        def type_pattern_match(value):
            if isinstance(value, str):
                return len(value)
            elif isinstance(value, int):
                return value
            else:
                return 0
        
        print(f"Type pattern matching (string): {type_pattern_match('hello')}")
        print(f"Type pattern matching (int): {type_pattern_match(42)}")
    
    # ============================================================================
    # 6. Case Classes → Dataclasses
    # ============================================================================
    
    def dataclass_examples(self):
        """Demonstrate dataclass conversions."""
        print("\n=== Dataclass Examples ===")
        
        # Define dataclasses (equivalent to Scala case classes)
        @dataclass
        class Person:
            """Equivalent to Scala case class Person(name: String, age: Int)."""
            name: str
            age: int
        
        @dataclass
        class Transaction:
            """Equivalent to Scala case class Transaction(id: Int, amount: Double, category: String)."""
            id: int
            amount: float
            category: str
            metadata: Dict[str, str] = field(default_factory=dict)
        
        # Create instances
        person = Person("Alice", 30)
        transaction = Transaction(1, 100.50, "Groceries")
        transaction.metadata = {"store": "Walmart"}
        
        print(f"Person: {person}")
        print(f"Transaction: {transaction}")
        
        # Access fields
        print(f"Person name: {person.name}")
        print(f"Transaction amount: {transaction.amount}")
    
    # ============================================================================
    # 7. Options / Null Safety
    # ============================================================================
    
    def option_examples(self):
        """Demonstrate Option/None handling."""
        print("\n=== Option/None Examples ===")
        
        # Scala: val maybeName: Option[String] = Some("Dan")
        # Python: Optional[str] with actual value or None
        maybe_name: Optional[str] = "Dan"
        
        # Scala: maybeName.getOrElse("Anonymous")
        name = maybe_name or "Anonymous"
        print(f"Option with value: {name}")
        
        # Scala: val maybeName: Option[String] = None
        maybe_name = None
        
        # Scala: maybeName.getOrElse("Anonymous")
        name = maybe_name or "Anonymous"
        print(f"Option with None: {name}")
        
        # More explicit None handling
        def safe_get_name(maybe_name: Optional[str]) -> str:
            if maybe_name is not None:
                return maybe_name
            else:
                return "Anonymous"
        
        print(f"Safe get (None): {safe_get_name(None)}")
        print(f"Safe get (value): {safe_get_name('John')}")
    
    # ============================================================================
    # 8. Collection Methods
    # ============================================================================
    
    def collection_methods_examples(self):
        """Demonstrate collection method conversions."""
        print("\n=== Collection Methods Examples ===")
        
        numbers = [1, 2, 3, 4, 5]
        
        # Map
        # Scala: list.map(_ * 2)
        # Python: List comprehension or map()
        doubled = [x * 2 for x in numbers]
        print(f"Map (list comprehension): {doubled}")
        
        doubled_map = list(map(lambda x: x * 2, numbers))
        print(f"Map (map function): {doubled_map}")
        
        # Filter
        # Scala: list.filter(_ > 2)
        # Python: List comprehension or filter()
        filtered = [x for x in numbers if x > 2]
        print(f"Filter (list comprehension): {filtered}")
        
        filtered_func = list(filter(lambda x: x > 2, numbers))
        print(f"Filter (filter function): {filtered_func}")
        
        # FlatMap
        # Scala: list.flatMap(x => List(x, x * 2))
        # Python: Nested list comprehension
        flat_mapped = [y for x in numbers for y in [x, x * 2]]
        print(f"FlatMap: {flat_mapped}")
        
        # Reduce
        # Scala: list.reduce(_ + _)
        # Python: functools.reduce()
        total = functools.reduce(lambda x, y: x + y, numbers)
        print(f"Reduce (sum): {total}")
        
        # GroupBy
        # Scala: list.groupBy(_ % 2 == 0)
        # Python: defaultdict with loop
        words = ["apple", "banana", "cherry", "date", "elderberry"]
        grouped = defaultdict(list)
        for word in words:
            grouped[len(word)].append(word)
        print(f"GroupBy (by length): {dict(grouped)}")
    
    # ============================================================================
    # 9. Immutability / Functional Programming
    # ============================================================================
    
    def immutability_examples(self):
        """Demonstrate immutability patterns."""
        print("\n=== Immutability Examples ===")
        
        # Python doesn't have built-in immutability like Scala's val
        # But we can use conventions and frozen collections
        
        # Tuple (immutable)
        point = (3, 4)
        print(f"Immutable tuple: {point}")
        
        # NamedTuple (immutable)
        from typing import NamedTuple
        
        class Point(NamedTuple):
            x: int
            y: int
        
        p = Point(3, 4)
        print(f"Immutable NamedTuple: {p}")
        
        # Frozenset (immutable set)
        immutable_set = frozenset([1, 2, 3])
        print(f"Immutable set: {immutable_set}")
        
        # Functional programming with itertools
        # Scala: (1 to 5).toList.flatMap(x => List(x, x * 2))
        from itertools import chain
        numbers = range(1, 6)
        flat_result = list(chain.from_iterable([x, x * 2] for x in numbers))
        print(f"Functional flatMap: {flat_result}")
        
        # Pipeline-style processing
        # Scala: numbers.map(_ * 2).filter(_ > 5).sum
        result = sum(x * 2 for x in numbers if x * 2 > 5)
        print(f"Pipeline processing: {result}")
    
    # ============================================================================
    # 10. Advanced Patterns
    # ============================================================================
    
    def advanced_patterns_examples(self):
        """Demonstrate advanced Scala patterns in Python."""
        print("\n=== Advanced Patterns Examples ===")
        
        # Sealed traits → Enum
        # Scala: sealed trait Status; case object Active extends Status
        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"
            PENDING = "pending"
        
        status = Status.ACTIVE
        print(f"Enum pattern: {status}")
        
        # Companion objects → Class methods
        # Scala: object StringUtils { def reverse(s: String): String = s.reverse }
        class StringUtils:
            @staticmethod
            def reverse(s: str) -> str:
                return s[::-1]
        
        result = StringUtils.reverse("hello")
        print(f"Static method: {result}")
        
        # Trait → Abstract base class
        # Scala: trait Printable { def print(): Unit }
        from abc import ABC, abstractmethod
        
        class Printable(ABC):
            @abstractmethod
            def print_info(self) -> None:
                pass
        
        class Document(Printable):
            def __init__(self, content: str):
                self.content = content
            
            def print_info(self) -> None:
                print(f"Document: {self.content}")
        
        doc = Document("Hello World")
        doc.print_info()
    
    # ============================================================================
    # 11. Top Challenges in Scala → Python Conversion
    # ============================================================================
    
    def challenge_1_immutability_vs_mutability(self):
        """Challenge 1: Immutability vs Mutability."""
        print("\n=== Challenge 1: Immutability vs Mutability ===")
        
        # Scala: val x = 5 (immutable by default)
        # Python: Must explicitly handle immutability
        x = 5  # Mutable by default
        print(f"Python variable (mutable): {x}")
        
        # Python alternatives for immutability
        # 1. Tuple (immutable)
        point = (3, 4)
        print(f"Immutable tuple: {point}")
        
        # 2. NamedTuple (immutable)
        from typing import NamedTuple
        class Point(NamedTuple):
            x: int
            y: int
        
        p = Point(3, 4)
        print(f"Immutable NamedTuple: {p}")
        
        # 3. Frozen dataclass (Python 3.7+)
        @dataclass(frozen=True)
        class ImmutablePerson:
            name: str
            age: int
        
        person = ImmutablePerson("Alice", 30)
        print(f"Immutable dataclass: {person}")
        
        # 4. Frozenset (immutable set)
        immutable_set = frozenset([1, 2, 3])
        print(f"Immutable set: {immutable_set}")
    
    def challenge_2_strong_typing_and_generics(self):
        """Challenge 2: Strong Typing and Generics."""
        print("\n=== Challenge 2: Strong Typing and Generics ===")
        
        # Scala: Strong static typing with type inference
        # Python: Dynamic typing (unless using mypy or type hints)
        
        # Type hints (Python 3.5+)
        def add(a: int, b: int) -> int:
            return a + b
        
        # Generic types
        from typing import List, Dict, Optional, TypeVar, Generic
        
        T = TypeVar('T')
        
        class Box(Generic[T]):
            def __init__(self, item: T):
                self.item = item
            
            def get(self) -> T:
                return self.item
        
        # Usage with type hints
        int_box = Box[int](42)
        str_box = Box[str]("hello")
        
        print(f"Generic Box (int): {int_box.get()}")
        print(f"Generic Box (str): {str_box.get()}")
        
        # Type checking with mypy (runtime example)
        def process_list(items: List[int]) -> int:
            return sum(items)
        
        result = process_list([1, 2, 3, 4, 5])
        print(f"Typed function result: {result}")
    
    def challenge_3_option_either_try_monads(self):
        """Challenge 3: Option / Either / Try Monads."""
        print("\n=== Challenge 3: Option / Either / Try Monads ===")
        
        # Scala: Option, Either, Try to avoid null
        # Python: Use None, try/except, or third-party libraries
        
        # Option → Optional
        # Scala: val name = user.name.getOrElse("Anonymous")
        # Python: name = user.name if user.name is not None else "Anonymous"
        
        class User:
            def __init__(self, name: Optional[str] = None):
                self.name = name
        
        user1 = User("Dan")
        user2 = User(None)
        
        name1 = user1.name if user1.name is not None else "Anonymous"
        name2 = user2.name if user2.name is not None else "Anonymous"
        
        print(f"User with name: {name1}")
        print(f"User without name: {name2}")
        
        # Either → Union types or custom implementation
        from typing import Union, Literal
        
        # Scala: Either[String, Int]
        # Python: Union[str, int] or Literal
        Result = Union[str, int]
        
        def divide(a: int, b: int) -> Result:
            if b == 0:
                return "Division by zero"
            else:
                return a // b
        
        result1 = divide(10, 2)
        result2 = divide(10, 0)
        
        print(f"Division success: {result1}")
        print(f"Division error: {result2}")
        
        # Try → try/except
        # Scala: Try { riskyOperation() }
        # Python: try/except
        
        def risky_operation() -> Union[str, int]:
            try:
                # Simulate risky operation
                import random
                if random.random() > 0.5:
                    return 42
                else:
                    raise ValueError("Operation failed")
            except Exception as e:
                return f"Error: {e}"
        
        result = risky_operation()
        print(f"Risky operation result: {result}")
    
    def challenge_4_pattern_matching(self):
        """Challenge 4: Pattern Matching."""
        print("\n=== Challenge 4: Pattern Matching ===")
        
        # Scala: Powerful and concise with match-case and deconstruction
        # Python < 3.10: No pattern matching
        # Python ≥ 3.10: Has structural match, but still limited
        
        # Python 3.10+ pattern matching
        def analyze_shape(shape):
            match shape:
                case (x, y):
                    return f"Point at ({x}, {y})"
                case {"type": "circle", "radius": r}:
                    return f"Circle with radius {r}"
                case {"type": "rectangle", "width": w, "height": h}:
                    return f"Rectangle {w}x{h}"
                case _:
                    return "Unknown shape"
        
        # Test pattern matching
        point = (3, 4)
        circle = {"type": "circle", "radius": 5}
        rectangle = {"type": "rectangle", "width": 10, "height": 20}
        
        print(f"Point analysis: {analyze_shape(point)}")
        print(f"Circle analysis: {analyze_shape(circle)}")
        print(f"Rectangle analysis: {analyze_shape(rectangle)}")
        
        # Python < 3.10 workaround
        def analyze_shape_legacy(shape):
            if isinstance(shape, tuple) and len(shape) == 2:
                x, y = shape
                return f"Point at ({x}, {y})"
            elif isinstance(shape, dict):
                if shape.get("type") == "circle":
                    return f"Circle with radius {shape.get('radius')}"
                elif shape.get("type") == "rectangle":
                    return f"Rectangle {shape.get('width')}x{shape.get('height')}"
            return "Unknown shape"
        
        print(f"Legacy point analysis: {analyze_shape_legacy(point)}")
    
    def challenge_5_functional_idioms(self):
        """Challenge 5: Functional Idioms."""
        print("\n=== Challenge 5: Functional Idioms ===")
        
        # Scala: First-class functions, immutability, combinators
        # Python: Imperative-first. List comprehensions cover map, filter, zip
        
        numbers = [1, 2, 3, 4, 5]
        
        # Scala: list.foldLeft(0)(_ + _)
        # Python: functools.reduce
        total = functools.reduce(lambda acc, x: acc + x, numbers, 0)
        print(f"Fold/reduce: {total}")
        
        # Scala: list.scanLeft(0)(_ + _)
        # Python: Manual implementation
        def scan_left(func, initial, items):
            result = [initial]
            for item in items:
                result.append(func(result[-1], item))
            return result
        
        scan_result = scan_left(lambda acc, x: acc + x, 0, numbers)
        print(f"Scan left: {scan_result}")
        
        # Scala: list.zipWithIndex
        # Python: enumerate
        indexed = list(enumerate(numbers))
        print(f"Zip with index: {indexed}")
        
        # Scala: list.partition(_ % 2 == 0)
        # Python: Manual implementation
        evens, odds = [], []
        for x in numbers:
            if x % 2 == 0:
                evens.append(x)
            else:
                odds.append(x)
        print(f"Partition: evens={evens}, odds={odds}")
        
        # Scala: list.collect { case x if x > 2 => x * 2 }
        # Python: List comprehension with condition
        collected = [x * 2 for x in numbers if x > 2]
        print(f"Collect: {collected}")
    
    def challenge_6_for_expressions_comprehensions(self):
        """Challenge 6: For-Expressions / Comprehensions."""
        print("\n=== Challenge 6: For-Expressions / Comprehensions ===")
        
        # Scala: for { x <- List(1, 2, 3); y <- List(4, 5) } yield (x, y)
        # Python: [(x, y) for x in [1, 2, 3] for y in [4, 5]]
        
        # Simple comprehension
        result = [(x, y) for x in [1, 2, 3] for y in [4, 5]]
        print(f"Cross product: {result}")
        
        # With conditions
        filtered = [(x, y) for x in [1, 2, 3] for y in [4, 5] if x + y > 5]
        print(f"Filtered cross product: {filtered}")
        
        # Nested comprehensions (can get hard to read)
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        flattened = [item for row in matrix for item in row]
        print(f"Flattened matrix: {flattened}")
        
        # More complex example
        users = ["Alice", "Bob", "Charlie"]
        permissions = ["read", "write", "admin"]
        user_perms = [(user, perm) for user in users for perm in permissions if user != "Charlie" or perm != "admin"]
        print(f"User permissions: {user_perms}")
    
    def challenge_7_concurrency_models(self):
        """Challenge 7: Concurrency Models (Futures / Actors / Streams)."""
        print("\n=== Challenge 7: Concurrency Models ===")
        
        # Scala: Futures, Akka Actors, reactive streams
        # Python: AsyncIO, async/await, concurrent.futures, threading, multiprocessing
        
        import asyncio
        import concurrent.futures
        import threading
        import time
        
        # Scala: Future { expensiveOperation() }
        # Python: concurrent.futures.ThreadPoolExecutor
        
        def expensive_operation(name: str) -> str:
            time.sleep(1)  # Simulate work
            return f"Result from {name}"
        
        # Thread-based futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(expensive_operation, f"task_{i}")
                for i in range(3)
            ]
            results = [future.result() for future in futures]
        
        print(f"Thread-based futures: {results}")
        
        # AsyncIO (Python's modern async approach)
        async def async_expensive_operation(name: str) -> str:
            await asyncio.sleep(1)  # Simulate async work
            return f"Async result from {name}"
        
        async def run_async_operations():
            tasks = [
                async_expensive_operation(f"async_task_{i}")
                for i in range(3)
            ]
            return await asyncio.gather(*tasks)
        
        # Run async operations
        async_results = asyncio.run(run_async_operations())
        print(f"AsyncIO results: {async_results}")
        
        # Actor pattern simulation (simplified)
        class SimpleActor:
            def __init__(self, name: str):
                self.name = name
                self.messages = []
            
            def send(self, message: str):
                self.messages.append(message)
                print(f"Actor {self.name} received: {message}")
        
        actor1 = SimpleActor("Actor1")
        actor2 = SimpleActor("Actor2")
        
        actor1.send("Hello from Actor2")
        actor2.send("Hello from Actor1")
    
    def challenge_8_object_oriented_traits_mixins(self):
        """Challenge 8: Object-Oriented + Traits / Mixins."""
        print("\n=== Challenge 8: Object-Oriented + Traits / Mixins ===")
        
        # Scala: Supports traits for multiple inheritance and clean mixins
        # Python: Supports mixins but with MRO (Method Resolution Order) issues
        
        # Scala: trait Logger { def log(msg: String): Unit }
        # Python: Mixin class
        
        class Logger:
            def log(self, msg: str) -> None:
                print(f"LOG: {msg}")
        
        class Database:
            def save(self, data: str) -> None:
                print(f"SAVING: {data}")
        
        # Multiple inheritance with mixins
        class App(Logger, Database):
            def process(self, data: str) -> None:
                self.log(f"Processing {data}")
                self.save(data)
        
        app = App()
        app.process("user data")
        
        # Method Resolution Order (MRO)
        print(f"App MRO: {App.__mro__}")
        
        # Trait-like behavior with abstract base classes
        from abc import ABC, abstractmethod
        
        class Service(ABC):
            @abstractmethod
            def execute(self) -> str:
                pass
        
        class EmailService(Service):
            def execute(self) -> str:
                return "Email sent"
        
        class SMSService(Service):
            def execute(self) -> str:
                return "SMS sent"
        
        services = [EmailService(), SMSService()]
        for service in services:
            print(f"Service result: {service.execute()}")
    
    def challenge_9_collections_advanced_ops(self):
        """Challenge 9: Collections: groupBy, zipWithIndex, sliding, etc."""
        print("\n=== Challenge 9: Collections: Advanced Operations ===")
        
        # Scala: list.zipWithIndex, list.sliding(2).toList, list.groupBy(_ % 2)
        # Python: Manual implementations or itertools
        
        from itertools import islice
        
        numbers = [1, 2, 3, 4, 5, 6]
        
        # zipWithIndex
        # Scala: list.zipWithIndex
        # Python: list(enumerate(lst))
        indexed = list(enumerate(numbers))
        print(f"Zip with index: {indexed}")
        
        # sliding (window)
        # Scala: list.sliding(2).toList
        # Python: Manual implementation
        def sliding(lst, size):
            return [lst[i:i+size] for i in range(len(lst) - size + 1)]
        
        windows = sliding(numbers, 2)
        print(f"Sliding windows (size 2): {windows}")
        
        # groupBy
        # Scala: list.groupBy(_ % 2)
        # Python: defaultdict
        grouped = defaultdict(list)
        for x in numbers:
            grouped[x % 2].append(x)
        print(f"Grouped by even/odd: {dict(grouped)}")
        
        # partition
        # Scala: list.partition(_ > 3)
        # Python: Manual implementation
        def partition(lst, predicate):
            left, right = [], []
            for item in lst:
                if predicate(item):
                    right.append(item)
                else:
                    left.append(item)
            return left, right
        
        left, right = partition(numbers, lambda x: x > 3)
        print(f"Partition (>3): left={left}, right={right}")
        
        # distinct
        # Scala: list.distinct
        # Python: list(set(lst)) or dict.fromkeys(lst)
        duplicates = [1, 2, 2, 3, 3, 4]
        distinct = list(dict.fromkeys(duplicates))
        print(f"Distinct: {distinct}")
        
        # take/drop
        # Scala: list.take(3), list.drop(2)
        # Python: lst[:3], lst[2:]
        taken = numbers[:3]
        dropped = numbers[2:]
        print(f"Take 3: {taken}")
        print(f"Drop 2: {dropped}")
    
    def challenge_10_custom_extractors_unapply(self):
        """Challenge 10: Custom Extractors and unapply."""
        print("\n=== Challenge 10: Custom Extractors and unapply ===")
        
        # Scala: Custom extractors with unapply
        # Python: No native destructuring with validation
        
        # Scala: object Email { def unapply(str: String): Option[(String, String)] = ... }
        # Python: Custom parser or regex
        
        import re
        
        class EmailExtractor:
            @staticmethod
            def unapply(email: str) -> Optional[tuple[str, str]]:
                pattern = r'^([^@]+)@([^@]+)$'
                match = re.match(pattern, email)
                if match:
                    return match.groups()
                return None
        
        # Usage
        extractor = EmailExtractor()
        
        test_emails = ["dan@gmail.com", "invalid-email", "user@domain.org"]
        
        for email in test_emails:
            result = extractor.unapply(email)
            if result:
                user, domain = result
                print(f"Valid email: {user}@{domain}")
            else:
                print(f"Invalid email: {email}")
        
        # Alternative: Using dataclasses for structured data
        @dataclass
        class Email:
            user: str
            domain: str
            
            @classmethod
            def parse(cls, email_str: str) -> Optional['Email']:
                pattern = r'^([^@]+)@([^@]+)$'
                match = re.match(pattern, email_str)
                if match:
                    user, domain = match.groups()
                    return cls(user, domain)
                return None
        
        # Usage with dataclass
        for email_str in test_emails:
            email = Email.parse(email_str)
            if email:
                print(f"Parsed email: {email}")
            else:
                print(f"Failed to parse: {email_str}")
    
    def run_all_challenges(self):
        """Run all conversion challenges."""
        print("\n" + "=" * 60)
        print("TOP CHALLENGES IN SCALA → PYTHON CONVERSION")
        print("=" * 60)
        
        self.challenge_1_immutability_vs_mutability()
        self.challenge_2_strong_typing_and_generics()
        self.challenge_3_option_either_try_monads()
        self.challenge_4_pattern_matching()
        self.challenge_5_functional_idioms()
        self.challenge_6_for_expressions_comprehensions()
        self.challenge_7_concurrency_models()
        self.challenge_8_object_oriented_traits_mixins()
        self.challenge_9_collections_advanced_ops()
        self.challenge_10_custom_extractors_unapply()
        
        print("\n" + "=" * 60)
        print("All conversion challenges completed!")
        print("=" * 60)
    
    def run_all_examples(self):
        """Run all conversion examples."""
        print("Scala to Python Conversion Examples")
        print("=" * 50)
        
        self.variable_declaration_examples()
        self.control_flow_examples()
        self.function_examples()
        self.collection_examples()
        self.pattern_matching_examples()
        self.dataclass_examples()
        self.option_examples()
        self.collection_methods_examples()
        self.immutability_examples()
        self.advanced_patterns_examples()
        
        print("\n" + "=" * 50)
        print("All conversion examples completed!")
        
        # Run the top challenges
        self.run_all_challenges()


# ============================================================================
# Practical Financial Data Examples
# ============================================================================

@dataclass
class FinancialTransaction:
    """Equivalent to Scala case class for financial transactions."""
    id: int
    amount: float
    category: str
    date: str
    metadata: Dict[str, str] = field(default_factory=dict)


class FinancialDataProcessor:
    """
    Demonstrates Scala-to-Python conversion in a financial context.
    
    This class shows how to convert Scala financial processing patterns
    to Python while maintaining the same business logic.
    """
    
    def __init__(self):
        self.transactions = [
            FinancialTransaction(1, -120.50, "Groceries", "2024-07-01"),
            FinancialTransaction(2, -75.20, "Transportation", "2024-07-03"),
            FinancialTransaction(3, 5000.00, "Salary", "2024-07-05"),
            FinancialTransaction(4, -45.30, "Entertainment", "2024-07-07"),
        ]
    
    def process_transactions_scala_style(self):
        """
        Demonstrate Scala-style processing converted to Python.
        
        Scala equivalent:
        transactions
          .filter(_.amount < 0)
          .groupBy(_.category)
          .mapValues(_.map(_.amount).sum)
        """
        print("\n=== Financial Data Processing (Scala-style) ===")
        
        # Filter negative amounts (expenses)
        expenses = [tx for tx in self.transactions if tx.amount < 0]
        print(f"Expenses: {expenses}")
        
        # Group by category and sum amounts
        category_totals = defaultdict(float)
        for tx in expenses:
            category_totals[tx.category] += abs(tx.amount)
        
        print(f"Category totals: {dict(category_totals)}")
        
        # Functional style with reduce
        total_expenses = functools.reduce(
            lambda acc, tx: acc + abs(tx.amount) if tx.amount < 0 else acc,
            self.transactions,
            0.0
        )
        print(f"Total expenses: ${total_expenses:.2f}")
    
    def pattern_matching_financial_data(self):
        """Demonstrate pattern matching for financial data."""
        print("\n=== Financial Pattern Matching ===")
        
        def categorize_transaction(tx: FinancialTransaction) -> str:
            """Pattern matching equivalent for transaction categorization."""
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
        
        for tx in self.transactions:
            category = categorize_transaction(tx)
            print(f"Transaction {tx.id}: {category} - ${tx.amount:.2f}")


if __name__ == "__main__":
    # Run all conversion examples
    converter = ConversionExamples()
    converter.run_all_examples()
    
    # Run financial data examples
    processor = FinancialDataProcessor()
    processor.process_transactions_scala_style()
    processor.pattern_matching_financial_data() 