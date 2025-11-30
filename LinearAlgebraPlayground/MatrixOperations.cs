using MathNet.Numerics.LinearAlgebra.Complex;
using Complex = System.Numerics.Complex;

namespace LinearAlgebraPlayground
{
    /// <summary>
    /// Provides operations for creating and manipulating matrices.
    /// Used for quantum gate representations and linear transformations.
    /// </summary>
    public static class MatrixOperations
    {
        /// <summary>
        /// Creates a 2×2 real-valued matrix and a 2D real-valued vector.
        /// Accepts a 2D array for the matrix and a 1D array for the vector.
        /// 
        /// Matrix-vector multiplication represents linear transformations - a fundamental concept
        /// in quantum mechanics where quantum gates (unitary matrices) transform quantum states (vectors).
        /// 
        /// The 'out' parameters allow this method to return both the matrix and vector for
        /// subsequent transformation operations.
        /// </summary>
        /// <param name="matrixValues">2D array of values for the 2×2 matrix</param>
        /// <param name="vectorValues">Array of 2 values for the vector</param>
        /// <param name="matrix">Output: constructed 2×2 matrix</param>
        /// <param name="vector">Output: constructed 2D vector</param>
        public static void Create2x2MatrixAndVector(
            double[,] matrixValues,
            double[] vectorValues,
            out MathNet.Numerics.LinearAlgebra.Double.DenseMatrix matrix,
            out MathNet.Numerics.LinearAlgebra.Double.DenseVector vector)
        {
            matrix = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(matrixValues);
            vector = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray(vectorValues);
        }

        /// <summary>
        /// Applies a matrix transformation to a vector through matrix-vector multiplication.
        /// This operation represents how quantum gates transform quantum states.
        /// 
        /// Matrix-vector multiplication process:
        /// For a 2×2 matrix A and 2D vector x:
        /// A * x computes each component as the dot product of matrix rows with the vector.
        /// 
        /// Example: A = [ 1  2 ]  and x = [ 1 ]
        ///              [ 3  4 ]          [ 0 ]
        /// 
        /// A * x = [ 1*1 + 2*0 ]   [ 1 ]
        ///         [ 3*1 + 4*0 ] = [ 3 ]
        /// 
        /// In quantum computing:
        /// - The matrix A represents a quantum gate (unitary transformation)
        /// - The vector x represents the quantum state
        /// - The result is the transformed quantum state after applying the gate
        /// </summary>
        public static MathNet.Numerics.LinearAlgebra.Double.DenseVector ApplyTransformation(
            MathNet.Numerics.LinearAlgebra.Double.DenseMatrix matrix,
            MathNet.Numerics.LinearAlgebra.Double.DenseVector vector)
        {
            return matrix * vector;
        }

        /// <summary>
        /// Computes the Kronecker product (tensor product) of two matrices.
        /// Essential for constructing multi-qubit quantum gates from single-qubit gates.
        /// 
        /// For matrices A (size m×n) and B (size p×q), the Kronecker product A ⊗ B
        /// is an (mp)×(nq) matrix where each element A[i,j] is replaced by the
        /// scalar multiple A[i,j]·B.
        /// 
        /// Example: For H ⊗ I (Hadamard on first qubit, Identity on second):
        /// Creates a 4×4 matrix that applies H to the first qubit while leaving
        /// the second qubit unchanged.
        /// </summary>
        public static DenseMatrix Kronecker(DenseMatrix A, DenseMatrix B)
        {
            int rA = A.RowCount, cA = A.ColumnCount;
            int rB = B.RowCount, cB = B.ColumnCount;

            var result = DenseMatrix.Create(rA * rB, cA * cB, Complex.Zero);

            for (int i = 0; i < rA; i++)
            {
                for (int j = 0; j < cA; j++)
                {
                    var scalar = A[i, j];
                    for (int k = 0; k < rB; k++)
                    {
                        for (int l = 0; l < cB; l++)
                        {
                            result[i * rB + k, j * cB + l] = scalar * B[k, l];
                        }
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Checks if a matrix is unitary.
        /// A unitary matrix U satisfies U†U = I (conjugate transpose times matrix equals identity).
        /// 
        /// Unitary matrices are essential in quantum computing because:
        /// - They represent reversible quantum operations
        /// - They preserve the normalization of quantum states
        /// - They preserve inner products (distances between states)
        /// 
        /// All valid quantum gates must be unitary matrices.
        /// </summary>
        public static bool IsUnitary(DenseMatrix U)
        {
            var UH = U.ConjugateTranspose();
            var product = UH * U;
            var I = DenseMatrix.Create(U.RowCount, U.ColumnCount, (i, j) => i == j ? Complex.One : Complex.Zero);

            double tol = 1e-9;
            for (int i = 0; i < U.RowCount; i++)
            {
                for (int j = 0; j < U.ColumnCount; j++)
                {
                    if ((product[i, j] - I[i, j]).Magnitude > tol) return false;
                }
            }
            return true;
        }
    }
}
