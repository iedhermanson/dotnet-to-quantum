using MathNet.Numerics.LinearAlgebra.Complex;
using Complex = System.Numerics.Complex;

namespace LinearAlgebraPlayground
{
    /// <summary>
    /// Provides operations for creating and manipulating vectors in both real and complex vector spaces.
    /// Used for quantum state representation and basic linear algebra operations.
    /// </summary>
    public static class VectorOperations
    {
        /// <summary>
        /// Creates two 3D real-valued vectors for demonstrating basic linear algebra operations.
        /// Accepts arrays of values to construct vectors in ℝ³ (3-dimensional real vector space).
        /// 
        /// Uses DenseVector from MathNet.Numerics - a concrete implementation that stores all elements
        /// in memory (vs sparse vectors that only store non-zero elements). Dense vectors are ideal
        /// for quantum computing where most amplitudes are non-zero.
        /// 
        /// The 'out' parameters allow this method to return multiple values - both vectors are
        /// created here and passed back to the caller for use in vector operations.
        /// </summary>
        /// <param name="vValues">Array of 3 values for vector v</param>
        /// <param name="wValues">Array of 3 values for vector w</param>
        /// <param name="v">Output: constructed vector v as DenseVector</param>
        /// <param name="w">Output: constructed vector w as DenseVector</param>
        public static void Create3DRealVectors(
            double[] vValues,
            double[] wValues,
            out MathNet.Numerics.LinearAlgebra.Double.DenseVector v,
            out MathNet.Numerics.LinearAlgebra.Double.DenseVector w)
        {
            v = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray(vValues);
            w = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray(wValues);
        }

        /// <summary>
        /// Creates two complex-valued vectors for demonstrating complex linear algebra operations.
        /// Accepts arrays of Complex values to construct vectors in ℂⁿ (complex vector space).
        /// 
        /// Complex vectors are essential in quantum mechanics where quantum states are represented
        /// as complex-valued vectors. Each component has both real and imaginary parts.
        /// 
        /// The 'out' parameters allow this method to return multiple values - both vectors are
        /// created here and passed back to the caller for quantum operations.
        /// </summary>
        /// <param name="vcValues">Array of Complex values for vector vc</param>
        /// <param name="wcValues">Array of Complex values for vector wc</param>
        /// <param name="vc">Output: constructed complex vector vc</param>
        /// <param name="wc">Output: constructed complex vector wc</param>
        public static void CreateComplexVectors(
            Complex[] vcValues,
            Complex[] wcValues,
            out DenseVector vc,
            out DenseVector wc)
        {
            vc = DenseVector.OfArray(vcValues);
            wc = DenseVector.OfArray(wcValues);
        }

        /// <summary>
        /// Demonstrates vector addition - a fundamental operation in linear algebra.
        /// Addition is performed component-wise: each element is added to its corresponding element.
        /// 
        /// For v = [1, 2, 3]ᵀ and w = [4, 5, 6]ᵀ:
        /// v + w = [1+4, 2+5, 3+6]ᵀ = [5, 7, 9]ᵀ
        /// 
        /// In quantum mechanics, vector addition represents superposition - combining quantum states.
        /// The '+' operator in C# is overloaded by MathNet.Numerics to perform element-wise addition.
        /// This operation preserves the vector space structure and is associative and commutative.
        /// </summary>
        public static MathNet.Numerics.LinearAlgebra.Double.DenseVector Add(
            MathNet.Numerics.LinearAlgebra.Double.DenseVector v,
            MathNet.Numerics.LinearAlgebra.Double.DenseVector w)
        {
            return v + w;
        }

        /// <summary>
        /// Demonstrates scalar multiplication - scaling a vector by a constant factor.
        /// Each component of the vector is multiplied by the scalar value.
        /// 
        /// For v = [1, 2, 3]ᵀ and scalar c = 2:
        /// c·v = 2·[1, 2, 3]ᵀ = [2·1, 2·2, 2·3]ᵀ = [2, 4, 6]ᵀ
        /// 
        /// In quantum mechanics, scalar multiplication changes the amplitude (probability magnitude)
        /// of a quantum state. When combined with normalization, it allows us to weight different
        /// basis states in a superposition. The '*' operator performs this scalar multiplication.
        /// </summary>
        public static MathNet.Numerics.LinearAlgebra.Double.DenseVector ScalarMultiply(
            MathNet.Numerics.LinearAlgebra.Double.DenseVector v,
            double scalar)
        {
            return scalar * v;
        }

        /// <summary>
        /// Demonstrates the dot product (inner product) for real vectors.
        /// The dot product is the sum of the products of corresponding components.
        /// 
        /// For v = [1, 2, 3]ᵀ and w = [4, 5, 6]ᵀ:
        /// v · w = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
        /// 
        /// The dot product measures how "aligned" two vectors are:
        /// - Returns 0 if vectors are orthogonal (perpendicular)
        /// - Returns positive if pointing in similar directions
        /// - Returns negative if pointing in opposite directions
        /// 
        /// In quantum mechanics, the inner product (complex version with conjugation) gives
        /// probability amplitudes. For real vectors, it's equivalent to the standard dot product.
        /// </summary>
        public static double DotProduct(
            MathNet.Numerics.LinearAlgebra.Double.DenseVector v,
            MathNet.Numerics.LinearAlgebra.Double.DenseVector w)
        {
            return v.DotProduct(w);
        }

        /// <summary>
        /// Computes the Hermitian inner product (complex conjugate dot product) of two complex vectors.
        /// This is the fundamental inner product in quantum mechanics, denoted in bra-ket notation as ⟨v|w⟩.
        /// 
        /// Process:
        /// 1. Take the complex conjugate of the first vector (flip sign of imaginary parts)
        /// 2. Perform element-wise multiplication with the second vector
        /// 3. Sum all the products
        /// 
        /// For vc = [1+0i, 0+1i] = [1, i] and wc = [2-1i, 3+4i]:
        /// ⟨vc|wc⟩ = conjugate(vc) · wc
        ///         = [1, -i] · [2-1i, 3+4i]
        ///         = (1)×(2-1i) + (-i)×(3+4i)
        ///         = (2-i) + (-3i-4i²)
        ///         = (2-i) + (-3i+4)
        ///         = 6-4i
        /// 
        /// This operation is fundamental to quantum mechanics:
        /// - Used to calculate probability amplitudes
        /// - Determines measurement probabilities via Born rule: P = |⟨ψ|φ⟩|²
        /// - Tests orthogonality of quantum states (⟨ψ|φ⟩ = 0 means states are distinguishable)
        /// </summary>
        public static Complex HermitianInnerProduct(DenseVector vc, DenseVector wc)
        {
            return vc.ConjugateDotProduct(wc);
        }
    }
}
