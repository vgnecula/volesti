import highspy
from scipy.io import mmwrite
from scipy.sparse import coo_matrix, csc_matrix
import numpy as np

mps = "rococoC10-001000.mps"
H = highspy.Highs()
H.readModel(mps)

# Get dimensions
m = H.getNumRow()  # Number of constraints (rows)
n = H.getNumCol()  # Number of variables (columns)
print(f"Matrix dimensions: {m} x {n}")

# Get the constraint matrix
lp = H.getLp()
A = lp.a_matrix_

# Extract matrix data
ptr = A.start_
ind = A.index_  # These are row indices
val = A.value_

print(f"Matrix format info:")
print(f"ptr length: {len(ptr)}")
print(f"ind length: {len(ind)}")
print(f"val length: {len(val)}")

# CSC format confirmed (column-wise storage)
print("Converting from CSC format...")
col = np.repeat(np.arange(n), np.diff(ptr))
row = ind
M = coo_matrix((val, (row, col)), shape=(m, n))

# Save matrix in Matrix Market format
mmwrite("rococoC10-001000.mtx", M)
print("Matrix saved to rococoC10-001000.mtx")

# Get constraint bounds (RHS)
try:
    row_lower = lp.row_lower_
    row_upper = lp.row_upper_
    np.save("row_lower.npy", row_lower)
    np.save("row_upper.npy", row_upper)
    print("Row bounds saved to row_lower.npy and row_upper.npy")
except AttributeError as e:
    print(f"Could not access row bounds: {e}")

# Get variable bounds
try:
    col_lower = lp.col_lower_
    col_upper = lp.col_upper_
    np.save("col_lower.npy", col_lower)
    np.save("col_upper.npy", col_upper)
    print("Variable bounds saved to col_lower.npy and col_upper.npy")
except AttributeError as e:
    print(f"Could not access column bounds: {e}")

# Get objective coefficients
try:
    obj_coeff = lp.col_cost_
    np.save("obj_coeff.npy", obj_coeff)
    print("Objective coefficients saved to obj_coeff.npy")
except AttributeError as e:
    print(f"Could not access objective coefficients: {e}")

print(f"\nMatrix statistics:")
print(f"Shape: {M.shape}")
print(f"Non-zeros: {M.nnz}")
print(f"Density: {M.nnz / (m * n):.6f}")