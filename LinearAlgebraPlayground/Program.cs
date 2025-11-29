using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using Complex = System.Numerics.Complex;

namespace LinearAlgebraPlayground
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("DotNet to Quantum: Linear Algebra Playground");
            Week1();
            Week2();
            Week3();
            Week4();
            Week5();
            // Week6 will be Q#, with comparison hooks here.
        }

        // Week 1: vectors, matrices, complex numbers, dot product
        static void Week1()
        {
            Console.WriteLine("\n--- Week 1: Foundations ---");
            RealVectorOperations();
            ComplexVectorInnerProduct();
            MatrixVectorTransformation();
        }

        /// <summary>
        /// Demonstrates basic vector operations: addition, scalar multiplication, and dot product.
        /// Vectors are fundamental building blocks - they represent states in quantum mechanics.
        /// </summary>
        static void RealVectorOperations()
        {
            // Create two 3D real-valued vectors
            Create3DRealVectors(
                [1, 2, 3],
                [4, 5, 6],
                out MathNet.Numerics.LinearAlgebra.Double.DenseVector v, 
                out MathNet.Numerics.LinearAlgebra.Double.DenseVector w);

            var vPlusW = VectorAddition(v, w);
            Console.WriteLine($"v + w = {vPlusW}");

            var scalarTimesV = ScalarMultiplication(v);
            Console.WriteLine($"2 * v = {scalarTimesV}");

            var dot_vw = DotProduct(v, w);
            Console.WriteLine($"dot(v, w) = {dot_vw}");
        }

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
        private static void Create3DRealVectors(
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
        private static void CreateComplexVectors(
            Complex[] vcValues,
            Complex[] wcValues,
            out DenseVector vc,
            out DenseVector wc)
        {
            // vc example: [1+0i, 0+1i] = [1, i]
            vc = DenseVector.OfArray(vcValues);
            
            // wc example: [2-1i, 3+4i]
            wc = DenseVector.OfArray(wcValues);
        }

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
        private static void Create2x2MatrixAndVector(
            double[,] matrixValues,
            double[] vectorValues,
            out MathNet.Numerics.LinearAlgebra.Double.DenseMatrix matrix,
            out MathNet.Numerics.LinearAlgebra.Double.DenseVector vector)
        {
            // Example matrix: A = [ 1  2 ]
            //                     [ 3  4 ]
            matrix = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(matrixValues);

            // Example vector: x = [1, 0]ᵀ (column vector)
            vector = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray(vectorValues);
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
        private static MathNet.Numerics.LinearAlgebra.Double.DenseVector VectorAddition(
            MathNet.Numerics.LinearAlgebra.Double.DenseVector v,
            MathNet.Numerics.LinearAlgebra.Double.DenseVector w)
        {
            // Vector addition: add corresponding components
            // [1,2,3] + [4,5,6] = [5,7,9]
            return (MathNet.Numerics.LinearAlgebra.Double.DenseVector)(v + w);
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
        private static MathNet.Numerics.LinearAlgebra.Double.DenseVector ScalarMultiplication(MathNet.Numerics.LinearAlgebra.Double.DenseVector v)
        {
            // Scalar multiplication: multiply each component by a scalar
            // 2 * [1,2,3] = [2,4,6]
            return (MathNet.Numerics.LinearAlgebra.Double.DenseVector)(2.0 * v);
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
        private static double DotProduct(
            MathNet.Numerics.LinearAlgebra.Double.DenseVector v,
            MathNet.Numerics.LinearAlgebra.Double.DenseVector w)
        {
            // Dot product: sum of products of corresponding components
            // [1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 32
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
        private static Complex HermitianInnerProduct(DenseVector vc, DenseVector wc)
        {
            // Complex inner product using bra-ket notation ⟨vc|wc⟩
            // Conjugates first vector: [1, -i] then dot product with [2-1i, 3+4i]
            // Result: (1)*(2-1i) + (-i)*(3+4i) = 2-i + (-3i+4) = 6-4i
            return vc.ConjugateDotProduct(wc);
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
        private static MathNet.Numerics.LinearAlgebra.Double.DenseVector ApplyMatrixTransformation(
            MathNet.Numerics.LinearAlgebra.Double.DenseMatrix matrix,
            MathNet.Numerics.LinearAlgebra.Double.DenseVector vector)
        {
            // Matrix-vector multiplication: A * x
            // [ 1  2 ] [ 1 ]   [ 1 ]
            // [ 3  4 ] [ 0 ] = [ 3 ]
            return (MathNet.Numerics.LinearAlgebra.Double.DenseVector)(matrix * vector);
        }

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
        private static void CreateBasisStates(out DenseVector ket0, out DenseVector ket1)
        {
            // |0⟩ = [1, 0]ᵀ - qubit in state "zero"
            ket0 = DenseVector.OfArray([Complex.One, Complex.Zero]);
            
            // |1⟩ = [0, 1]ᵀ - qubit in state "one"
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
        private static DenseVector CreateSuperposition(
            DenseVector basis1,
            DenseVector basis2,
            Complex alpha,
            Complex beta)
        {
            // Linear combination: |ψ⟩ = α|0⟩ + β|1⟩
            // Example: (1/√2)|0⟩ + (1/√2)|1⟩ creates equal superposition
            return (DenseVector)(alpha * basis1 + beta * basis2);
        }

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
        private static DenseMatrix CreatePauliXGate()
        {
            // Pauli-X gate (quantum NOT / bit-flip):
            // X = [ 0  1 ]    X|0⟩ = |1⟩
            //     [ 1  0 ]    X|1⟩ = |0⟩
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
        private static DenseMatrix CreatePauliZGate()
        {
            // Pauli-Z gate (phase-flip):
            // Z = [ 1   0 ]    Z|0⟩ = |0⟩
            //     [ 0  -1 ]    Z|1⟩ = -|1⟩
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
        private static DenseMatrix CreateHadamardGate()
        {
            // Hadamard gate (superposition creator):
            // H = (1/√2) [ 1   1 ]    H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩
            //            [ 1  -1 ]    H|1⟩ = (|0⟩ - |1⟩)/√2 = |-⟩
            return (Complex)(1.0 / Math.Sqrt(2)) * DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.One },
                { Complex.One, -Complex.One }
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
        private static DenseVector ApplyGateToState(DenseMatrix gate, DenseVector state)
        {
            // Matrix-vector multiplication: U|ψ⟩ = |ψ'⟩
            // Gate transforms the quantum state
            return (DenseVector)(gate * state);
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
        private static DenseMatrix CreateIdentityGate()
        {
            // Identity gate (do nothing):
            // I = [ 1  0 ]    I|0⟩ = |0⟩
            //     [ 0  1 ]    I|1⟩ = |1⟩
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
        private static DenseMatrix CreateCNOTGate()
        {
            // CNOT matrix with control qubit first, target qubit second:
            // [ 1  0  0  0 ]    |00⟩ → |00⟩ (control=0, do nothing)
            // [ 0  1  0  0 ]    |01⟩ → |01⟩ (control=0, do nothing)
            // [ 0  0  0  1 ]    |10⟩ → |11⟩ (control=1, flip target)
            // [ 0  0  1  0 ]    |11⟩ → |10⟩ (control=1, flip target)
            return DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero, Complex.Zero, Complex.Zero },
                { Complex.Zero, Complex.One, Complex.Zero, Complex.Zero },
                { Complex.Zero, Complex.Zero, Complex.Zero, Complex.One },
                { Complex.Zero, Complex.Zero, Complex.One, Complex.Zero }
            });
        }

        /// <summary>
        /// Demonstrates the complex inner product (Hermitian inner product).
        /// In quantum mechanics, the inner product ⟨v|w⟩ gives probability amplitudes.
        /// Uses conjugate of first vector: ⟨v|w⟩ = conjugate(v) · w
        /// </summary>
        static void ComplexVectorInnerProduct()
        {
            // Create two complex vectors
            CreateComplexVectors(
                [new Complex(1, 0), new Complex(0, 1)],
                [new Complex(2, -1), new Complex(3, 4)],
                out DenseVector vc,
                out DenseVector wc);

            var inner_vc_wc = HermitianInnerProduct(vc, wc);
            Console.WriteLine($"<v|w> (complex inner product) = {inner_vc_wc}");
        }

        /// <summary>
        /// Demonstrates a 2×2 matrix transformation applied to a 2D vector.
        /// Matrix-vector multiplication is how quantum gates transform quantum states.
        /// A|x⟩ transforms the state |x⟩ according to the linear map defined by matrix A.
        /// </summary>
        static void MatrixVectorTransformation()
        {
            // Create a 2×2 transformation matrix and a 2D vector
            Create2x2MatrixAndVector(
                new double[,] { {1, 2}, {3, 4} },
                [1, 0],
                out MathNet.Numerics.LinearAlgebra.Double.DenseMatrix A,
                out MathNet.Numerics.LinearAlgebra.Double.DenseVector x);

            var result = ApplyMatrixTransformation(A, x);
            Console.WriteLine($"A * x = {result}");
        }

        // Week 2: qubit states, normalization, probabilities
        static void Week2()
        {
            Console.WriteLine("\n--- Week 2: Quantum State Representation ---");
            ComputationalBasisStates();
            SuperpositionState();
        }

        /// <summary>
        /// Demonstrates the computational basis states |0⟩ and |1⟩.
        /// These are the "classical" states of a qubit, analogous to bits 0 and 1.
        /// |0⟩ = [1, 0]ᵀ and |1⟩ = [0, 1]ᵀ as complex column vectors.
        /// Normalized means the sum of squared magnitudes equals 1.
        /// </summary>
        static void ComputationalBasisStates()
        {
            // Create the computational basis states
            CreateBasisStates(out DenseVector ket0, out DenseVector ket1);

            Console.WriteLine($"|0> = {ket0}");
            Console.WriteLine($"|1> = {ket1}");

            // Verify normalization: |α₀|² + |α₁|² = 1
            // For valid quantum states, probabilities must sum to 1
            var ket0Normalized = IsNormalized(ket0);
            var ket1Normalized = IsNormalized(ket1);
            Console.WriteLine($"Is |0> normalized? {ket0Normalized}");
            Console.WriteLine($"Is |1> normalized? {ket1Normalized}");
        }

        /// <summary>
        /// Creates a superposition state: a linear combination of basis states.
        /// |ψ⟩ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1
        /// This is the "quantum" part - the qubit exists in both states simultaneously.
        /// Measurement collapses to |0⟩ with probability |α|² or |1⟩ with probability |β|².
        /// </summary>
        static void SuperpositionState()
        {
            CreateBasisStates(out DenseVector ket0, out DenseVector ket1);

            // Create equal superposition: |ψ⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
            // This is the |+⟩ state - 50% chance of measuring 0 or 1
            var alpha = new Complex(1.0/Math.Sqrt(2), 0);
            var beta  = new Complex(1.0/Math.Sqrt(2), 0);
            
            var psi = CreateSuperposition(ket0, ket1, alpha, beta);
            var psiNormalized = IsNormalized(psi);
            
            Console.WriteLine($"|psi> = {psi}, normalized? {psiNormalized}");

            // Calculate measurement probabilities in computational basis
            // P(0) = |⟨0|ψ⟩|² and P(1) = |⟨1|ψ⟩|²
            // Born rule: probability = squared magnitude of inner product
            var p0 = ProbabilityOfOutcome(psi, ket0);
            var p1 = ProbabilityOfOutcome(psi, ket1);
            Console.WriteLine($"P(0) = {p0:F4}, P(1) = {p1:F4} (sum={p0+p1:F4})");
        }

        // Week 3: gates as matrices, unitarity, apply gates
        static void Week3()
        {
            Console.WriteLine("\n--- Week 3: Quantum Gates as Matrices ---");
            DefineQuantumGates();
            ApplyHadamardGate();
        }

        /// <summary>
        /// Defines common single-qubit quantum gates as 2×2 unitary matrices.
        /// Quantum gates are reversible transformations (unitary: U†U = I).
        /// X gate: bit-flip (quantum NOT), swaps |0⟩ ↔ |1⟩
        /// Z gate: phase-flip, adds minus sign to |1⟩: |1⟩ → -|1⟩
        /// H gate: Hadamard, creates superposition from basis states
        /// </summary>
        static void DefineQuantumGates()
        {
            // Create the three fundamental single-qubit gates
            var X = CreatePauliXGate();
            var Z = CreatePauliZGate();
            var H = CreateHadamardGate();

            // Verify unitarity: U†U = I (unitary gates are reversible and preserve norm)
            var xUnitary = IsUnitary(X);
            var zUnitary = IsUnitary(Z);
            var hUnitary = IsUnitary(H);
            
            Console.WriteLine($"Is X unitary? {xUnitary}");
            Console.WriteLine($"Is Z unitary? {zUnitary}");
            Console.WriteLine($"Is H unitary? {hUnitary}");
        }

        /// <summary>
        /// Applies the Hadamard gate to create superposition.
        /// H|0⟩ = (1/√2)(|0⟩ + |1⟩) - equal superposition state |+⟩
        /// This is the most common way to create superposition in quantum algorithms.
        /// </summary>
        static void ApplyHadamardGate()
        {
            CreateBasisStates(out DenseVector ket0, out _);

            var H = CreateHadamardGate();

            // Apply H to |0⟩: transforms basis state into superposition
            // H|0⟩ = (1/√2)[1, 1]ᵀ ≈ [0.707, 0.707]ᵀ
            var result = ApplyGateToState(H, ket0);
            Console.WriteLine($"H|0> = {result}");
        }

        // Week 4: tensor products, 2-qubit states, CNOT
        static void Week4()
        {
            Console.WriteLine("\n--- Week 4: Multi-Qubit Systems ---");
            TwoQubitBasisStates();
            CNOTGateDefinition();
            CreateBellState();
        }

        /// <summary>
        /// Creates 2-qubit basis states using tensor product (Kronecker product).
        /// Tensor product ⊗ combines single-qubit states into multi-qubit states.
        /// For 2 qubits: 4 basis states |00⟩, |01⟩, |10⟩, |11⟩ (each a 4D vector).
        /// |ab⟩ = |a⟩ ⊗ |b⟩ where a, b ∈ {0, 1}
        /// </summary>
        static void TwoQubitBasisStates()
        {
            CreateBasisStates(out DenseVector ket0, out DenseVector ket1);

            // Create 2-qubit computational basis states via tensor product
            var ket00 = Tensor(ket0, ket0);
            var ket01 = Tensor(ket0, ket1);

            Console.WriteLine($"|00> = {ket00}");
            Console.WriteLine($"|01> = {ket01}");
        }

        /// <summary>
        /// Defines the CNOT (Controlled-NOT) gate - a 4×4 unitary matrix for 2 qubits.
        /// CNOT flips the target qubit (second) if control qubit (first) is |1⟩.
        /// Maps: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
        /// Essential for creating entanglement between qubits.
        /// </summary>
        static void CNOTGateDefinition()
        {
            var CNOT = CreateCNOTGate();
            Console.WriteLine("CNOT gate defined (4×4 matrix for 2-qubit operations)");
        }

        /// <summary>
        /// Creates a Bell state (maximally entangled 2-qubit state).
        /// Recipe: Start with |00⟩, apply (H ⊗ I), then apply CNOT.
        /// Result: |Φ+⟩ = (|00⟩ + |11⟩)/√2
        /// Bell states exhibit quantum entanglement - measuring one qubit instantly
        /// determines the other, regardless of distance (Einstein's "spooky action").
        /// </summary>
        static void CreateBellState()
        {
            CreateBasisStates(out DenseVector ket0, out _);
            var ket00 = Tensor(ket0, ket0);

            var H = CreateHadamardGate();
            var I2 = CreateIdentityGate();
            var CNOT = CreateCNOTGate();

            // Step 1: Apply (H ⊗ I) to |00⟩ → creates superposition on first qubit
            // (H ⊗ I)|00⟩ = (|0⟩ + |1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
            var HtensorI = Kronecker(H, I2);
            var bellPre = HtensorI * ket00;

            // Step 2: Apply CNOT → creates entanglement
            // CNOT(|00⟩ + |10⟩)/√2 = (|00⟩ + |11⟩)/√2 = |Φ+⟩ (Bell state)
            var bell = (DenseVector)(CNOT * bellPre);
            var bellNormalized = IsNormalized(bell);

            Console.WriteLine($"Bell state (Phi+): {bell}, normalized? {bellNormalized}");
        }

        // Week 5: measurement probabilities, eigenvalues/eigenvectors of Z
        static void Week5()
        {
            Console.WriteLine("\n--- Week 5: Measurement & Eigen Concepts ---");
            EigendecompositionOfPauliZ();
            MeasurementProbabilities();
        }

        /// <summary>
        /// Demonstrates eigenvalues and eigenvectors of the Pauli-Z gate.
        /// Eigenvector: A vector |v⟩ where A|v⟩ = λ|v⟩ (A scales |v⟩ by eigenvalue λ).
        /// For Z gate: |0⟩ is eigenvector with eigenvalue +1, |1⟩ with eigenvalue -1.
        /// Measurement outcomes correspond to eigenvalues of the observable operator.
        /// </summary>
        static void EigendecompositionOfPauliZ()
        {
            var Z = CreatePauliZGate();
            CreateBasisStates(out DenseVector ket0, out DenseVector ket1);

            // Z|0⟩ = +1|0⟩ (eigenvalue +1)
            var z0 = ApplyGateToState(Z, ket0);
            Console.WriteLine($"Z|0> = {z0}");
            
            // Z|1⟩ = -1|1⟩ (eigenvalue -1)
            var z1 = ApplyGateToState(Z, ket1);
            Console.WriteLine($"Z|1> = {z1}");
            
            Console.WriteLine($"Eigenvalues (expected): +1 for |0>, -1 for |1>");
        }

        /// <summary>
        /// Calculates measurement probabilities using the Born rule.
        /// Born rule: P(outcome i) = |⟨i|ψ⟩|² (squared magnitude of inner product).
        /// For a superposition state, probabilities are determined by the amplitudes.
        /// Probabilities must sum to 1 (normalization condition).
        /// </summary>
        static void MeasurementProbabilities()
        {
            CreateBasisStates(out DenseVector ket0, out DenseVector ket1);

            // Create equal superposition: |ψ⟩ = (|0⟩ + |1⟩)/√2
            // Each amplitude has magnitude 1/√2, so probability = (1/√2)² = 1/2
            var alpha = new Complex(1.0/Math.Sqrt(2), 0);
            var beta  = new Complex(1.0/Math.Sqrt(2), 0);
            var psi = CreateSuperposition(ket0, ket1, alpha, beta);

            // Calculate measurement probabilities in Z basis (computational basis)
            // P(0) = |⟨0|ψ⟩|² = |α|² = 1/2
            var p0 = ProbabilityOfOutcome(psi, ket0);
            
            // P(1) = |⟨1|ψ⟩|² = |β|² = 1/2
            var p1 = ProbabilityOfOutcome(psi, ket1);
            
            Console.WriteLine($"P_Z(0) = {p0:F4}, P_Z(1) = {p1:F4} (sum={p0+p1:F4})");
        }

        // Utilities

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
        static bool IsNormalized(Vector<Complex> vector)
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
        static double ProbabilityOfOutcome(Vector<Complex> state, Vector<Complex> basisVector)
        {
            var amplitude = state.ConjugateDotProduct(basisVector); // ⟨basis|state⟩
            return Math.Pow(amplitude.Magnitude, 2);
        }

        static DenseVector Tensor(DenseVector a, DenseVector b)
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

        static DenseMatrix Kronecker(DenseMatrix A, DenseMatrix B)
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

        static bool IsUnitary(DenseMatrix U)
        {
            var UH = U.ConjugateTranspose();
            var product = UH * U;
            // Compare with identity
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