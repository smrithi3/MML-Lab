import numpy as np
import matplotlib.pyplot as plt
ratings_matrix = np.array([
    [4, 5, 0, 3],
    [2, 0, 4, 5],
    [0, 3, 5, 0],
    [4, 0, 2, 0]
])
U, Sigma, Vt = np.linalg.svd(ratings_matrix, full_matrices=False)
plt.plot(Sigma)
plt.title('Singular Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
k = 2
U_k = U[:, :k]
Sigma_k = np.diag(Sigma[:k])
Vt_k = Vt[:k, :]
ratings_reconstructed = np.dot(U_k, np.dot(Sigma_k, Vt_k))
customer_factors = U_k
product_factors = Vt_k.T
print("Customer Factors:")
print(customer_factors)
print("\nProduct Factors:")
print(product_factors)
missing_row = 2
missing_col = 1
predicted_rating = np.dot(U_k[missing_row, :], np.dot(Sigma_k, Vt_k[:, missing_col]))
print(f"\nPredicted Rating for missing value at ({missing_row}, {missing_col}): {predicted_rating}")

