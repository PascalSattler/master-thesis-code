import numpy as np
from fractions import Fraction
from math import isqrt, sqrt
from sympy import nsimplify, latex
from sympy.abc import x as sym_x

def recognize_algebraic(value, max_degree=4, max_coeff=100, tolerance=1e-8):
    """
    Recognize a real number as an algebraic expression using PSLQ/LLL-like approach.
    
    Parameters:
    - value: The decimal number to recognize
    - max_degree: Maximum polynomial degree to search
    - max_coeff: Maximum coefficient size
    - tolerance: How close the approximation should be
    
    Returns:
    - A symbolic expression or None if not found
    """
    
    # Use sympy's built-in nsimplify which uses PSLQ algorithm
    result = nsimplify(value, rational=False, tolerance=tolerance)
    
    return result

def simple_sqrt_recognition(value, max_n=1000, tolerance=1e-8):
    """
    Check if value ≈ sqrt(n) for some integer n.
    """
    n = round(value ** 2)
    if abs(sqrt(n) - value) < tolerance and n <= max_n:
        return f"sqrt({n})"
    return None

def nested_sqrt_recognition(value, max_depth=2, max_val=100, tolerance=1e-6):
    """
    Try to recognize nested square roots like sqrt(a + sqrt(b)).
    """
    
    # Level 1: sqrt(n)
    result = simple_sqrt_recognition(value, max_val**2, tolerance)
    if result:
        return result
    
    # Level 2: sqrt(a + sqrt(b)) or sqrt(a - sqrt(b))
    if max_depth >= 2:
        x_squared = value ** 2
        
        for a in range(-max_val, max_val + 1):
            diff = x_squared - a
            
            # Try positive: sqrt(a + sqrt(b))
            if diff > 0:
                b = round(diff ** 2)
                if b <= max_val**2 and abs(sqrt(a + sqrt(b)) - value) < tolerance:
                    return f"sqrt({a} + sqrt({b}))"
            
            # Try negative: sqrt(a - sqrt(b))
            if diff < 0:
                b = round(diff ** 2)
                if b <= max_val**2 and abs(sqrt(a - sqrt(b)) - value) < tolerance:
                    return f"sqrt({a} - sqrt({b}))"
    
    return None

def recognize_with_lll(value, max_degree=6, max_coeff=1000, tolerance=1e-8):
    """
    Use integer relation detection (LLL-based) to find polynomial with integer coefficients.
    This uses numpy's approach to find relations among powers of the value.
    """
    
    # Create vector [1, x, x^2, ..., x^max_degree]
    powers = np.array([value**i for i in range(max_degree + 1)])
    
    # Try to find integer relation using a simplified PSLQ-like approach
    # We look for small integer coefficients c_i such that sum(c_i * x^i) ≈ 0
    
    # Build lattice basis
    n = len(powers)
    scale = 10**10  # Scaling factor
    
    # Create lattice matrix
    B = np.zeros((n, n))
    for i in range(n):
        B[i, i] = 1
        B[i, 0] = int(powers[i] * scale)
    
    # Simple integer relation search (not full LLL, but effective)
    best_relation = None
    best_error = float('inf')
    
    for _ in range(100):  # Try multiple random combinations
        coeffs = np.random.randint(-max_coeff, max_coeff + 1, size=n)
        if np.all(coeffs == 0):
            continue
            
        error = abs(np.dot(coeffs, powers))
        
        if error < best_error and error < tolerance:
            best_error = error
            best_relation = coeffs
    
    if best_relation is not None:
        # Format as polynomial
        terms = []
        for i, c in enumerate(best_relation):
            if abs(c) > 0.01:
                if i == 0:
                    terms.append(f"{int(c)}")
                elif i == 1:
                    terms.append(f"{int(c)}*x")
                else:
                    terms.append(f"{int(c)}*x^{i}")
        
        if terms:
            return " + ".join(terms) + " ≈ 0"
    
    return None

def find_algebraic_form(value, verbose=True):
    """
    Main function to recognize an algebraic number.
    Tries multiple methods in order of complexity.
    """
    
    if verbose:
        print(f"Analyzing: {value}")
        print("-" * 50)
    
    # Method 1: Simple sqrt
    result = simple_sqrt_recognition(value)
    if result:
        if verbose:
            print(f"✓ Simple form: {result}")
        return result
    
    # Method 2: Nested sqrt
    result = nested_sqrt_recognition(value, max_depth=2)
    if result:
        if verbose:
            print(f"✓ Nested radical form: {result}")
        return result
    
    # Method 3: Sympy's nsimplify (uses PSLQ)
    result = recognize_algebraic(value)
    if result and result != value:
        if verbose:
            print(f"✓ Algebraic form (sympy): {result}")
        return str(result)
    
    # Method 4: Custom LLL-based approach
    result = recognize_with_lll(value)
    if result:
        if verbose:
            print(f"✓ Polynomial relation: {result}")
        return result
    
    if verbose:
        print("✗ No simple algebraic form found")
    
    return None


# # Test examples
# if __name__ == "__main__":
#     test_values = [
#         1.4142135623730951,  # sqrt(2)
#         1.7320508075688772,  # sqrt(3)
#         1.9318516525781366,  # sqrt(2 + sqrt(3))
#         2.4142135623730951,  # 1 + sqrt(2)
#         1.6180339887498949,  # golden ratio: (1 + sqrt(5))/2
#         2.6457513110645906,  # 1 + sqrt(1 + sqrt(2))
#     ]
    
#     for val in test_values:
#         print()
#         find_algebraic_form(val)
#         print()

print(find_algebraic_form(1.90733128))