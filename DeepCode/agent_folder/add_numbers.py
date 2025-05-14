def add_numbers(a, b):
    """
    A function that adds two numbers and returns the result.
    
    Args:
        a: The first number
        b: The second number
        
    Returns:
        The sum of a and b
    """
    return a + b

# Example usage
if __name__ == "__main__":
    # Get input from user
    a = float(input("Enter the first number (a): "))
    b = float(input("Enter the second number (b): "))
    
    # Calculate the sum
    result = add_numbers(a, b)
    
    # Display the result
    print(f"The sum of {a} + {b} = {result}")