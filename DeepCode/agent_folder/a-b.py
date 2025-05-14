#!/usr/bin/env python3
"""
File: a-b.py
Description: A simple Python script that performs subtraction between two numbers.
"""

def subtract(a, b):
    """
    Subtracts b from a and returns the result.
    
    Args:
        a (number): The first number (minuend)
        b (number): The second number (subtrahend)
        
    Returns:
        number: The difference between a and b (a - b)
    """
    return a - b

def get_numeric_input(prompt):
    """
    Helper function to get valid numeric input from the user.
    
    Args:
        prompt (str): The input prompt to display to the user
        
    Returns:
        float: The validated numeric input
    """
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Error: Please enter a valid number.")

def main():
    """
    Main function that handles user interaction and displays results.
    """
    print("Subtraction Calculator (a - b)")
    print("-" * 30)
    
    a = get_numeric_input("Enter the value of a: ")
    b = get_numeric_input("Enter the value of b: ")
    
    result = subtract(a, b)
    print(f"\nResult: {a} - {b} = {result}")
    
    # Handle integer vs. float display format for cleaner output
    if result == int(result):
        print(f"\nResult: {a} - {b} = {int(result)}")
    else:
        print(f"\nResult: {a} - {b} = {result}")

if __name__ == "__main__":
    main()