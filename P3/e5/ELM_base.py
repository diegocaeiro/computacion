import random
import math
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import time

n_cores = 8

class ELM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights between input layer and hidden layer
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        print(self.weights_input_hidden)

        # Initialize biases of the hidden layer
        self.bias_hidden = [0] * hidden_size
    
    def train(self, X_train, y_train):
        # Calculate the output of the hidden layer
        
        t_train_hidden_output = time.time()
        hidden_output = self.sigmoid_matrix_multiply(X_train, self.weights_input_hidden, self.bias_hidden)
        t_train_hidden_output = time.time() - t_train_hidden_output

        # Calculate the weights of the output layer using the pseudo-inverse
        t_train_weights_hidden_output = time.time()
        self.weights_hidden_output = self.calculate_pseudo_inverse(hidden_output, y_train) # paralelizar
        t_train_weights_hidden_output = time.time() - t_train_weights_hidden_output

        print(f'Tiempo para train_hidden_output >> {t_train_hidden_output}')
        print(f'Tiempo para train_wieghts_hidden_output >> {t_train_weights_hidden_output}')
    
    def predict(self, X_test):

        # Calculate the output of the hidden layer for the test data

        t_predic_hidden_output = time.time()
        hidden_output = self.sigmoid_matrix_multiply(X_test, self.weights_input_hidden, self.bias_hidden)
        t_predic_hidden_output = time.time() - t_predic_hidden_output
        
        # Calculate the output of the output layer
        t_predic_output = time.time()
        output = self.matrix_multiply(hidden_output, self.weights_hidden_output) # paralelizar
        t_predic_output = time.time() - t_predic_output

        # Convert the output to a list of lists of sparse matrices
        # Flatten the list comprehension and convert it to a NumPy array
        t_predict_y_pred_array = time.time()
        y_pred_array = np.array([sparse_matrix.toarray().flatten() for row in output for sparse_matrix in row])
        t_predict_y_pred_array = time.time() - t_predict_y_pred_array

        # Reshape the flattened array to have one row per prediction
        predicted_classes = np.argmax(y_pred_array, axis=1)

        print(f'Tiempo para predict_hidden_output >> {t_predic_hidden_output}')
        print(f'Tiempo para predic_output >> {t_predic_output}')
        print(f'Tiempo para predict_y_pred_array >> {t_predict_y_pred_array}')
    

        return predicted_classes
    
    def sigmoid(self, x):
            try: 
                return 1 / (1 + math.exp(-x))
            except:
                if x < 0:
                    return 0
                elif x > 0:
                    return 1
    
    def sigmoid_matrix_multiply(self, X, W, b):
        # Calculate the net input to the hidden layer
        net_hidden = [[sum(X[i][j] * W[j][k] for j in range(self.input_size)) + b[k] for k in range(self.hidden_size)] for i in range(len(X))]
        # Apply the sigmoid function to the net input
        return [[self.sigmoid(net_hidden[i][j]) for j in range(self.hidden_size)] for i in range(len(X))]
    
    def calculate_pseudo_inverse(self, X, Y):
        # Calculate the Moore-Penrose pseudo-inverse of the input matrix
        X_transpose = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
        pseudo_inverse = self.matrix_multiply(self.matrix_inverse(self.matrix_multiply(X_transpose, X)), X_transpose) # llama a matrix_multiply
        
        # Multiply the pseudo-inverse with the target matrix
        return self.matrix_multiply(pseudo_inverse, Y)
    
    def matrix_multiply(self, A, B):
        # Matrix multiplication
        rows_A = len(A)
        cols_A = len(A[0])
        try:
            rows_B = len(B) 
        except:
           rows_B = B.shape[0]  
        try:
            cols_B = len(B[0])
        except:
           cols_B = B[0].shape[0]
        
        if cols_A != rows_B:
            raise ValueError("Number of columns in A must be equal to number of rows in B")
        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    def matrix_inverse(self, A):
        # Calculate the inverse of a square matrix using Gaussian elimination
        n = len(A)
        A_inv = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
        for col in range(n):
            # Find pivot
            max_row = max(range(col, n), key=lambda i: abs(A[i][col]))
            A[col], A[max_row] = A[max_row], A[col]
            A_inv[col], A_inv[max_row] = A_inv[max_row], A_inv[col]
            # Make diagonal element 1
            diag = A[col][col]
            for j in range(n):
                A[col][j] /= diag
                A_inv[col][j] /= diag
            # Zero out other elements in the column
            for i in range(n):
                if i != col:
                    ratio = A[i][col]
                    for j in range(n):
                        A[i][j] -= ratio * A[col][j]
                        A_inv[i][j] -= ratio * A_inv[col][j]
        return A_inv


class ELMMP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights between input layer and hidden layer
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_input_hidden = [[-0.9785912039022524, 0.1854937638690366, -0.5894576143474242, -0.668128251339497, -0.27372905570625305, 0.04071060359100831, 0.7592525938837091, -0.6147878606837092, 0.8693905057259672, 0.189115886918938], [0.03402119552647975, 0.4468603171446337, -0.9995786785947898, 0.2324852044355994, 0.7233752055444123, -0.7472140852557625, -0.777937695069987, 0.7110068662594542, -0.00289888421594231, 0.19638279880846876], [0.5274035098757974, 0.33507833165223766, 0.6481121494285387, 0.9124151347954472, -0.5745695372829143, -0.45162280126886833, 0.26981988788376166, 0.17620226084770985, -0.863956518576557, -0.188786785056003], [0.4049531305609384, 0.5677078543593608, 0.08617216238802095, -0.1793671231223093, -0.2113770240234114, 0.7506566250414894, -0.493894794574419, -0.5683528357919434, 0.10146646640979906, 0.6703636605162226]]
        # Initialize biases of the hidden layer
        self.bias_hidden = [0] * hidden_size
    
    def train(self, X_train, y_train):
        # Calculate the output of the hidden layer
        
        t_train_hidden_output = time.time()
        hidden_output = self.sigmoid_matrix_multiply(X_train, self.weights_input_hidden, self.bias_hidden)
        t_train_hidden_output = time.time() - t_train_hidden_output

        # Calculate the weights of the output layer using the pseudo-inverse
        t_train_weights_hidden_output = time.time()
        self.weights_hidden_output = self.calculate_pseudo_inverse(hidden_output, y_train) # paralelizar
        t_train_weights_hidden_output = time.time() - t_train_weights_hidden_output

        print(f'Tiempo para train_hidden_output >> {t_train_hidden_output}')
        print(f'Tiempo para train_wieghts_hidden_output >> {t_train_weights_hidden_output}')
    
    def predict(self, X_test):

        # Calculate the output of the hidden layer for the test data

        t_predic_hidden_output = time.time()
        hidden_output = self.sigmoid_matrix_multiply(X_test, self.weights_input_hidden, self.bias_hidden)
        t_predic_hidden_output = time.time() - t_predic_hidden_output
        
        # Calculate the output of the output layer
        t_predic_output = time.time()
        output = self.matrix_multiply(hidden_output, self.weights_hidden_output) # paralelizar
        t_predic_output = time.time() - t_predic_output

        # Convert the output to a list of lists of sparse matrices
        # Flatten the list comprehension and convert it to a NumPy array
        t_predict_y_pred_array = time.time()
        y_pred_array = np.array([sparse_matrix.toarray().flatten() for row in output for sparse_matrix in row])
        t_predict_y_pred_array = time.time() - t_predict_y_pred_array

        # Reshape the flattened array to have one row per prediction
        predicted_classes = np.argmax(y_pred_array, axis=1)

        print(f'Tiempo para predict_hidden_output >> {t_predic_hidden_output}')
        print(f'Tiempo para predic_output >> {t_predic_output}')
        print(f'Tiempo para predict_y_pred_array >> {t_predict_y_pred_array}')
    

        return predicted_classes
    
    def sigmoid(self, x):
            try: 
                return 1 / (1 + math.exp(-x))
            except:
                if x < 0:
                    return 0
                elif x > 0:
                    return 1
    
    def sigmoid_matrix_multiply(self, X, W, b):
        # Calculate the net input to the hidden layer
        net_hidden = [[sum(X[i][j] * W[j][k] for j in range(self.input_size)) + b[k] for k in range(self.hidden_size)] for i in range(len(X))]
        # Apply the sigmoid function to the net input
        return [[self.sigmoid(net_hidden[i][j]) for j in range(self.hidden_size)] for i in range(len(X))]
    
    
    def calculate_pseudo_inverse(self, X, Y):
        # Calculate the Moore-Penrose pseudo-inverse of the input matrix
        X_transpose = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]

        pseudo_inverse = self.matrix_multiply(self.matrix_inverse(self.matrix_multiply(X_transpose, X)), X_transpose)
        
        # Multiply the pseudo-inverse with the target matrix
        return self.matrix_multiply(pseudo_inverse, Y)

    def matrix_multiply(self, A, B):
        # Matrix multiplication

        rows_A = len(A)
        cols_A = len(A[0])

        try:
            rows_B = len(B) 
        except:
           rows_B = B.shape[0]  
        try:
            cols_B = len(B[0])
        except:
           cols_B = B[0].shape[0]

        if cols_A != rows_B:
            raise ValueError("Number of columns in A must be equal to number of rows in B")

        with Pool() as pool:
            result = pool.starmap(self.vector_multiply, [(A[i], B, cols_B) for i in range(len(A))])
            
        return result
    
    def vector_multiply(self, v1, B, cols_B):
        result = [sum(v1[i] * B[i][j] for i in range(len(v1))) for j in range(cols_B)]
        return result

    def matrix_inverse(self, A):
        # Calculate the inverse of a square matrix using Gaussian elimination
        n = len(A)
        A_inv = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
        for col in range(n):
            # Find pivot
            max_row = max(range(col, n), key=lambda i: abs(A[i][col]))
            A[col], A[max_row] = A[max_row], A[col]
            A_inv[col], A_inv[max_row] = A_inv[max_row], A_inv[col]
            # Make diagonal element 1
            diag = A[col][col]
            for j in range(n):
                A[col][j] /= diag
                A_inv[col][j] /= diag
            # Zero out other elements in the column
            for i in range(n):
                if i != col:
                    ratio = A[i][col]
                    for j in range(n):
                        A[i][j] -= ratio * A[col][j]
                        A_inv[i][j] -= ratio * A_inv[col][j]
        return A_inv
