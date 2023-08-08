import numpy as np
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'
def test_error_function(error):
    result = error(10, 5)
    expected = 5
    if result == expected:
        print(GREEN + "Test passed!"+ RESET)
    elif result == -5:
        print(BLUE + BOLD+"The implementation should be: "+ RESET+RESET+GREEN+"y - y_hat"+RESET)
    else:
        print(RED+"Error function test failed. Hint: Call the experts."+RESET)

def test_error_square_function(error_square):
    result = error_square(5)
    expected = 25
    if result == expected:
        print(GREEN+"Test passed!"+RESET)
    elif result == 10:
        print(BLUE+"You are tying to multiply by using only single * operator."+RESET)
        print(GREEN+BOLD+"Try using ** operator to square the error."+RESET+RESET)
    else:
        print(RED+"Error function test failed. Hint: Call the experts."+RESET)

def test_total_squared_error_function(total_squared_error):
    errors = [1, 4, 9, 16, 25]
    result = 0
    for i in range(len(errors)):
        result += total_squared_error(errors[i],1)
    expected = 55
    if result == expected:
        print(GREEN+"Test passed!"+RESET)
    else:
        print(RED+"Total squared error function test failed. Hint: Check your loop and summation logic."+RESET)

def test_mse_function(mse):
    result = mse(55, 5)
    expected = 5
    if result == expected:
        print(GREEN+ "Test passed!"+RESET)
    else:
        print(RED+"MSE function test failed. Hint: Verify your division and parameter order."+RESET)

def test_predicted_value(predicted_value):
    result = predicted_value(2, 3, 1)
    expected = 7
    if result == expected:
        print(GREEN + "Test passed!" + RESET)
    else:
        print(RED + "predicted_value test failed. Hint: Check your implementation." + RESET)

def test_compute_cost(compute_cost):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    w = 0.5
    b = 1.0
    result = compute_cost(x, y, w, b)
    expected = 1.375
    epsilon = 1e-6  # A small tolerance for comparison

    if abs(result - expected) < epsilon:
        print(GREEN + "Test passed!" + RESET)
    else:
        print(RED + "compute_cost test failed. Hint: Check your implementation of squared_error and total_squared_error." + RESET)

def test_compute_gradient(compute_gradient):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    w = 0.5
    b = 1.0
    result = compute_gradient(x, y, w, b)
    expected = (-4.7, -1.5)
    epsilon = 1e-6

    if abs(result[0] - expected[0]) < epsilon and abs(result[1] - expected[1]) < epsilon:
        print(GREEN + "Test passed!" + RESET)
    else:
        print(RED + "compute_gradient test failed. Hint: Check your implementation of squared_error and total_squared_error." + RESET)

def test_gradient_descent(gradient_descent, compute_cost, compute_gradient):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    learning_rate = 0.01
    num_iterations = 10
    
    w_final, b_final = gradient_descent(x, y, learning_rate, num_iterations)
    cost_final = compute_cost(x, y, w_final, b_final)
    
    print(f'Final parameters: w = {w_final:.4f}, b = {b_final:.4f}')
    print(f'Final cost: {cost_final:.6f}')
    
    expected_w = 0.7955439965516822  
    expected_b = 0.2544737290710181
    epsilon = 0.005  
    
    if abs(w_final - expected_w) < epsilon and abs(b_final - expected_b) < epsilon:
        print(GREEN + "Test passed!" + RESET)
    else:
        print(RED + "gradient_descent test failed. Hint: Check your implementation." + RESET)