using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using Complex = System.Numerics.Complex;

namespace LinearAlgebraPlayground
{
    /// <summary>
    /// Provides methods for creating and manipulating quantum states.
    /// Handles basis states, superposition states, and multi-qubit states.
    /// </summary>
    public static class QuantumStates
    {
        /// <summary>
        /// Creates the computational basis states |0⟩ and |1⟩ for a single qubit.
        /// These are the fundamental basis vectors of the 2-dimensional complex Hilbert space ℂ².
        /// 
        /// |0⟩ = [1, 0]ᵀ - qubit in state "zero" (analogous to classical bit 0)
        /// |1⟩ = [0, 1]ᵀ - qubit in state "one" (analogous to classical bit 1)
        /// 
        /// These states form an orthonormal basis:
        /// - ⟨0|0⟩ = 1, ⟨1|1⟩ = 1 (normalized)
        /// - ⟨0|1⟩ = 0, ⟨1|0⟩ = 0 (orthogonal)
        /// 
        /// Any single-qubit state can be expressed as a superposition: |ψ⟩ = α|0⟩ + β|1⟩
        /// where α and β are complex amplitudes satisfying |α|² + |β|² = 1.
        /// 
        /// The 'out' parameters return both basis states for use in quantum operations.
        /// </summary>
        /// <param name="ket0">Output: the |0⟩ basis state</param>
        /// <param name="ket1">Output: the |1⟩ basis state</param>
        public static void CreateBasisStates(out DenseVector ket0, out DenseVector ket1)
        {
            ket0 = DenseVector.OfArray([Complex.One, Complex.Zero]);
            ket1 = DenseVector.OfArray([Complex.Zero, Complex.One]);
        }

        /// <summary>
        /// Creates a superposition state by linearly combining two basis states with complex amplitudes.
        /// This is the fundamental quantum operation that distinguishes quantum from classical computation.
        /// 
        /// Computes: |ψ⟩ = α|basis1⟩ + β|basis2⟩
        /// 
        /// For a valid quantum state, amplitudes must satisfy the normalization condition:
        /// |α|² + |β|² = 1
        /// 
        /// This ensures that measurement probabilities sum to 1 (conservation of probability).
        /// 
        /// Physical interpretation:
        /// - α and β are probability amplitudes (complex numbers)
        /// - |α|² gives the probability of measuring the system in state |basis1⟩
        /// - |β|² gives the probability of measuring the system in state |basis2⟩
        /// - Before measurement, the qubit exists in both states simultaneously (superposition)
        /// - Measurement collapses the state to one of the basis states
        /// 
        /// Example: Equal superposition |+⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
        /// Results in 50% probability of measuring 0 or 1.
        /// </summary>
        /// <param name="basis1">First basis state (typically |0⟩)</param>
        /// <param name="basis2">Second basis state (typically |1⟩)</param>
        /// <param name="alpha">Complex amplitude for first basis state</param>
        /// <param name="beta">Complex amplitude for second basis state</param>
        /// <returns>The superposition state |ψ⟩ = α|basis1⟩ + β|basis2⟩</returns>
        public static DenseVector CreateSuperposition(
            DenseVector basis1,
            DenseVector basis2,
            Complex alpha,
            Complex beta)
        {
            return (DenseVector)(alpha * basis1 + beta * basis2);
        }

        /// <summary>
        /// Computes the tensor product (Kronecker product) of two quantum state vectors.
        /// Essential for constructing multi-qubit states from single-qubit states.
        /// 
        /// For vectors a (size m) and b (size n), the tensor product a ⊗ b
        /// is a vector of size (m×n) constructed by multiplying each element
        /// of a by the entire vector b.
        /// 
        /// Example: |0⟩ ⊗ |1⟩ = [1,0]ᵀ ⊗ [0,1]ᵀ = [0,1,0,0]ᵀ = |01⟩
        /// 
        /// This operation is how we build multi-qubit states:
        /// - |00⟩ = |0⟩ ⊗ |0⟩
        /// - |01⟩ = |0⟩ ⊗ |1⟩
        /// - |10⟩ = |1⟩ ⊗ |0⟩
        /// - |11⟩ = |1⟩ ⊗ |1⟩
        /// </summary>
        public static DenseVector Tensor(DenseVector a, DenseVector b)
        {
            var result = new Complex[a.Count * b.Count];
            int idx = 0;
            for (int i = 0; i < a.Count; i++)
            {
                for (int j = 0; j < b.Count; j++)
                {
                    result[idx++] = a[i] * b[j];
                }
            }
            return DenseVector.OfArray(result);
        }

        /// <summary>
        /// Checks if a quantum state vector is normalized (has unit length).
        /// A normalized state satisfies: Σᵢ |αᵢ|² = 1
        /// 
        /// This is a fundamental requirement in quantum mechanics:
        /// - The squared magnitudes of amplitudes represent probabilities
        /// - Probabilities must sum to 1 (conservation of probability)
        /// - Non-normalized states are not physically meaningful
        /// 
        /// Calculation process:
        /// 1. For each component αᵢ of the vector
        /// 2. Compute |αᵢ|² (magnitude squared)
        /// 3. Sum all |αᵢ|²
        /// 4. Check if sum ≈ 1 (within numerical tolerance)
        /// 
        /// Example: For |ψ⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
        /// Sum = |1/√2|² + |1/√2|² = 1/2 + 1/2 = 1 ✓
        /// </summary>
        /// <param name="vector">The quantum state vector to check</param>
        /// <returns>True if normalized (sum ≈ 1), false otherwise</returns>
        public static bool IsNormalized(Vector<Complex> vector)
        {
            double sum = 0.0;
            for (int i = 0; i < vector.Count; i++)
            {
                sum += Math.Pow(vector[i].Magnitude, 2);
            }
            return Math.Abs(sum - 1.0) < 1e-9;
        }

        /// <summary>
        /// Calculates the probability of measuring a quantum state in a specific basis state.
        /// Implements the Born rule: P = |⟨basis|state⟩|²
        /// 
        /// The Born rule is the fundamental postulate connecting quantum mechanics to measurement:
        /// - Take the inner product of the basis state and the quantum state
        /// - Square the magnitude of this amplitude to get probability
        /// 
        /// Process:
        /// 1. Compute ⟨basis|state⟩ (Hermitian inner product)
        /// 2. Take |⟨basis|state⟩|² (magnitude squared)
        /// 3. Result is probability of measuring the system in |basis⟩ state
        /// 
        /// Example: For |ψ⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
        /// P(0) = |⟨0|ψ⟩|² = |1/√2|² = 1/2 = 50%
        /// P(1) = |⟨1|ψ⟩|² = |1/√2|² = 1/2 = 50%
        /// </summary>
        /// <param name="state">The quantum state |ψ⟩ to measure</param>
        /// <param name="basisVector">The basis state |basis⟩ to measure in</param>
        /// <returns>Probability (0 to 1) of measuring the state in the given basis</returns>
        public static double ProbabilityOfOutcome(Vector<Complex> state, Vector<Complex> basisVector)
        {
            var amplitude = state.ConjugateDotProduct(basisVector);
            return Math.Pow(amplitude.Magnitude, 2);
        }
    }
}
