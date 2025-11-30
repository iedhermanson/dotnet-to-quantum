using MathNet.Numerics.LinearAlgebra.Complex;
using Complex = System.Numerics.Complex;

namespace LinearAlgebraPlayground
{
    /// <summary>
    /// Provides factory methods for creating standard quantum gates.
    /// All gates are unitary matrices that represent reversible quantum operations.
    /// </summary>
    public static class QuantumGates
    {
        /// <summary>
        /// Creates the Pauli-X gate (quantum NOT / bit-flip gate).
        /// The X gate is a 2×2 unitary matrix that flips qubit states.
        /// 
        /// Matrix form:
        /// X = [ 0  1 ]
        ///     [ 1  0 ]
        /// 
        /// Effect on basis states:
        /// X|0⟩ = |1⟩  (flips 0 to 1)
        /// X|1⟩ = |0⟩  (flips 1 to 0)
        /// 
        /// This is the quantum analog of the classical NOT gate, but it operates
        /// on quantum superpositions. For example:
        /// X(α|0⟩ + β|1⟩) = α|1⟩ + β|0⟩ (swaps amplitudes)
        /// 
        /// Properties:
        /// - Hermitian: X† = X (self-adjoint)
        /// - Unitary: X†X = I (preserves inner products)
        /// - Involutory: X² = I (applying twice returns to original state)
        /// - Eigenvalues: ±1
        /// 
        /// The X gate is one of the Pauli gates and is fundamental in quantum error correction.
        /// </summary>
        /// <returns>The Pauli-X gate as a 2×2 complex matrix</returns>
        public static DenseMatrix CreatePauliX()
        {
            return DenseMatrix.OfArray(new Complex[,] {
                { Complex.Zero, Complex.One },
                { Complex.One, Complex.Zero }
            });
        }

        /// <summary>
        /// Creates the Pauli-Z gate (phase-flip gate).
        /// The Z gate is a 2×2 unitary matrix that adds a phase to the |1⟩ component.
        /// 
        /// Matrix form:
        /// Z = [ 1   0 ]
        ///     [ 0  -1 ]
        /// 
        /// Effect on basis states:
        /// Z|0⟩ = |0⟩   (leaves |0⟩ unchanged)
        /// Z|1⟩ = -|1⟩  (adds -1 phase to |1⟩)
        /// 
        /// For superposition states:
        /// Z(α|0⟩ + β|1⟩) = α|0⟩ - β|1⟩ (negates the |1⟩ amplitude)
        /// 
        /// Properties:
        /// - Hermitian: Z† = Z (self-adjoint)
        /// - Unitary: Z†Z = I (preserves inner products)
        /// - Involutory: Z² = I (applying twice returns to original state)
        /// - Diagonal: eigenvalues +1 and -1 are directly visible
        /// - Eigenvectors: |0⟩ (eigenvalue +1) and |1⟩ (eigenvalue -1)
        /// 
        /// The Z gate represents measurement in the computational basis and is fundamental
        /// for phase kickback in quantum algorithms like phase estimation.
        /// </summary>
        /// <returns>The Pauli-Z gate as a 2×2 complex matrix</returns>
        public static DenseMatrix CreatePauliZ()
        {
            return DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero },
                { Complex.Zero, -Complex.One }
            });
        }

        /// <summary>
        /// Creates the Hadamard gate (superposition creator).
        /// The H gate is a 2×2 unitary matrix that creates equal superposition from basis states.
        /// 
        /// Matrix form:
        /// H = (1/√2) [ 1   1 ]
        ///            [ 1  -1 ]
        /// 
        /// Effect on basis states:
        /// H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩  (equal superposition with positive phase)
        /// H|1⟩ = (|0⟩ - |1⟩)/√2 = |-⟩  (equal superposition with negative phase)
        /// 
        /// The Hadamard gate is arguably the most important gate in quantum computing:
        /// - Creates superposition from computational basis states
        /// - Used at the start of most quantum algorithms
        /// - Transforms between computational basis {|0⟩, |1⟩} and Hadamard basis {|+⟩, |-⟩}
        /// - Self-inverse: H² = I (applying twice returns to original state)
        /// 
        /// Properties:
        /// - Hermitian: H† = H (self-adjoint)
        /// - Unitary: H†H = I (preserves inner products)
        /// - Symmetric: H = Hᵀ (matrix equals its transpose)
        /// - Involutory: H² = I
        /// 
        /// The factor 1/√2 ensures normalization - probabilities sum to 1.
        /// </summary>
        /// <returns>The Hadamard gate as a 2×2 complex matrix</returns>
        public static DenseMatrix CreateHadamard()
        {
            return (Complex)(1.0 / Math.Sqrt(2)) * DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.One },
                { Complex.One, -Complex.One }
            });
        }

        /// <summary>
        /// Creates the Identity gate (no-op gate).
        /// The I gate is a 2×2 unitary matrix that leaves any quantum state unchanged.
        /// 
        /// Matrix form:
        /// I = [ 1  0 ]
        ///     [ 0  1 ]
        /// 
        /// Effect on basis states:
        /// I|0⟩ = |0⟩  (leaves |0⟩ unchanged)
        /// I|1⟩ = |1⟩  (leaves |1⟩ unchanged)
        /// 
        /// For any state |ψ⟩:
        /// I|ψ⟩ = |ψ⟩  (identity operation)
        /// 
        /// Properties:
        /// - Hermitian: I† = I (self-adjoint)
        /// - Unitary: I†I = I (preserves inner products)
        /// - Diagonal: eigenvalues are all +1
        /// - Universal identity: IA = AI = A for any matrix A
        /// 
        /// The Identity gate is essential in multi-qubit systems:
        /// - Used in tensor products like H ⊗ I (apply H to first qubit, leave second unchanged)
        /// - Represents "do nothing" to specific qubits in multi-qubit operations
        /// - Forms the basis for defining unitarity: U†U = I
        /// </summary>
        /// <returns>The Identity gate as a 2×2 complex matrix</returns>
        public static DenseMatrix CreateIdentity()
        {
            return DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero },
                { Complex.Zero, Complex.One }
            });
        }

        /// <summary>
        /// Creates the CNOT (Controlled-NOT) gate.
        /// The CNOT gate is a 4×4 unitary matrix for 2-qubit operations that performs a controlled bit-flip.
        /// 
        /// Matrix form (control qubit first, target qubit second):
        /// CNOT = [ 1  0  0  0 ]
        ///        [ 0  1  0  0 ]
        ///        [ 0  0  0  1 ]
        ///        [ 0  0  1  0 ]
        /// 
        /// Effect on 2-qubit basis states:
        /// CNOT|00⟩ = |00⟩  (control=0, target unchanged)
        /// CNOT|01⟩ = |01⟩  (control=0, target unchanged)
        /// CNOT|10⟩ = |11⟩  (control=1, flip target: 0→1)
        /// CNOT|11⟩ = |10⟩  (control=1, flip target: 1→0)
        /// 
        /// The CNOT gate is THE fundamental two-qubit gate:
        /// - Creates entanglement between qubits
        /// - Essential building block for universal quantum computation
        /// - Together with single-qubit gates, forms a universal gate set
        /// - Used to create Bell states (maximally entangled states)
        /// 
        /// For superposition states:
        /// CNOT(α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩) = α|00⟩ + β|01⟩ + γ|11⟩ + δ|10⟩
        /// 
        /// Properties:
        /// - Hermitian: CNOT† = CNOT (self-adjoint)
        /// - Unitary: CNOT†·CNOT = I (preserves inner products)
        /// - Involutory: CNOT² = I (applying twice returns to original state)
        /// - Classical behavior on basis states (deterministic mapping)
        /// - Non-classical behavior on superpositions (creates entanglement)
        /// 
        /// Key quantum phenomenon:
        /// Starting from |00⟩, applying (H⊗I) then CNOT creates the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2,
        /// demonstrating quantum entanglement where measuring one qubit instantly determines the other.
        /// </summary>
        /// <returns>The CNOT gate as a 4×4 complex matrix</returns>
        public static DenseMatrix CreateCNOT()
        {
            return DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero, Complex.Zero, Complex.Zero },
                { Complex.Zero, Complex.One, Complex.Zero, Complex.Zero },
                { Complex.Zero, Complex.Zero, Complex.Zero, Complex.One },
                { Complex.Zero, Complex.Zero, Complex.One, Complex.Zero }
            });
        }

        /// <summary>
        /// Applies a quantum gate (unitary matrix) to a quantum state (vector).
        /// This is matrix-vector multiplication where the gate transforms the state.
        /// 
        /// Operation: |ψ'⟩ = U|ψ⟩
        /// 
        /// Where:
        /// - U is a unitary gate (preserves norm)
        /// - |ψ⟩ is the input quantum state
        /// - |ψ'⟩ is the output quantum state after transformation
        /// 
        /// Process:
        /// For each component of the output vector:
        /// ψ'ᵢ = Σⱼ Uᵢⱼψⱼ (sum over row i of gate times state vector)
        /// 
        /// Example: Hadamard gate on |0⟩
        /// H|0⟩ = (1/√2)[ 1   1 ][ 1 ]   (1/√2)[ 1 ]   (|0⟩ + |1⟩)/√2
        ///              [ 1  -1 ][ 0 ] =       [ 1 ] =
        /// 
        /// Key principles:
        /// - Unitary gates preserve normalization: if ⟨ψ|ψ⟩ = 1, then ⟨ψ'|ψ'⟩ = 1
        /// - Gates are reversible: can always compute U†|ψ'⟩ to recover |ψ⟩
        /// - Superposition is preserved and transformed
        /// </summary>
        /// <param name="gate">The quantum gate (unitary matrix) to apply</param>
        /// <param name="state">The quantum state vector to transform</param>
        /// <returns>The transformed quantum state after applying the gate</returns>
        public static DenseVector Apply(DenseMatrix gate, DenseVector state)
        {
            return gate * state;
        }
    }
}
