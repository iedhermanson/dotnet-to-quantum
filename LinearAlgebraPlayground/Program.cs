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
            var v = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray([1, 2, 3]);
            var w = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray([4, 5, 6]);

            // Vector addition: add corresponding components
            // [1,2,3] + [4,5,6] = [5,7,9]
            var vPlusW = v + w;

            // Scalar multiplication: multiply each component by a scalar
            // 2 * [1,2,3] = [2,4,6]
            var scalarTimesV = 2.0 * v;

            // Dot product: sum of products of corresponding components
            // [1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 32
            var dot_vw = v.DotProduct(w);

            Console.WriteLine($"v + w = {vPlusW}");
            Console.WriteLine($"2 * v = {scalarTimesV}");
            Console.WriteLine($"dot(v, w) = {dot_vw}");
        }

        /// <summary>
        /// Demonstrates the complex inner product (Hermitian inner product).
        /// In quantum mechanics, the inner product ⟨v|w⟩ gives probability amplitudes.
        /// Uses conjugate of first vector: ⟨v|w⟩ = conjugate(v) · w
        /// </summary>
        static void ComplexVectorInnerProduct()
        {
            // vc = [1+0i, 0+1i] = [1, i]
            var vc = DenseVector.OfArray([new Complex(1, 0), new Complex(0, 1)]);
            
            // wc = [2-1i, 3+4i]
            var wc = DenseVector.OfArray([new Complex(2, -1), new Complex(3, 4)]);

            // Complex inner product using bra-ket notation ⟨vc|wc⟩
            // Conjugates first vector: [1, -i] then dot product with [2-1i, 3+4i]
            // Result: (1)*(2-1i) + (-i)*(3+4i) = 2-i + (-3i+4) = 6-4i
            var inner_vc_wc = vc.ConjugateDotProduct(wc);
            Console.WriteLine($"<v|w> (complex inner product) = {inner_vc_wc}");
        }

        /// <summary>
        /// Demonstrates a 2×2 matrix transformation applied to a 2D vector.
        /// Matrix-vector multiplication is how quantum gates transform quantum states.
        /// A|x⟩ transforms the state |x⟩ according to the linear map defined by matrix A.
        /// </summary>
        static void MatrixVectorTransformation()
        {
            // Define a 2×2 transformation matrix:
            // A = [ 1  2 ]
            //     [ 3  4 ]
            var A = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(new double[,] {
                {1, 2},
                {3, 4}
            });

            // Define a 2D vector: x = [1, 0]ᵀ (column vector)
            var x = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray([1, 0]);

            // Apply the transformation: A * x
            // [ 1  2 ] [ 1 ]   [ 1*1 + 2*0 ]   [ 1 ]
            // [ 3  4 ] [ 0 ] = [ 3*1 + 4*0 ] = [ 3 ]
            Console.WriteLine($"A * x = {A * x}");
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
            // |0⟩ = [1, 0]ᵀ - qubit in state "zero"
            var ket0 = DenseVector.OfArray([Complex.One, Complex.Zero]);
            
            // |1⟩ = [0, 1]ᵀ - qubit in state "one"
            var ket1 = DenseVector.OfArray([Complex.Zero, Complex.One]);

            Console.WriteLine($"|0> = {ket0}");
            Console.WriteLine($"|1> = {ket1}");

            // Verify normalization: |α₀|² + |α₁|² = 1
            // For valid quantum states, probabilities must sum to 1
            Console.WriteLine($"Is |0> normalized? {IsNormalized(ket0)}");
            Console.WriteLine($"Is |1> normalized? {IsNormalized(ket1)}");
        }

        /// <summary>
        /// Creates a superposition state: a linear combination of basis states.
        /// |ψ⟩ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1
        /// This is the "quantum" part - the qubit exists in both states simultaneously.
        /// Measurement collapses to |0⟩ with probability |α|² or |1⟩ with probability |β|².
        /// </summary>
        static void SuperpositionState()
        {
            var ket0 = DenseVector.OfArray([Complex.One, Complex.Zero]);
            var ket1 = DenseVector.OfArray([Complex.Zero, Complex.One]);

            // Create equal superposition: |ψ⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
            // This is the |+⟩ state - 50% chance of measuring 0 or 1
            var alpha = new Complex(1.0/Math.Sqrt(2), 0);
            var beta  = new Complex(1.0/Math.Sqrt(2), 0);
            var psi = alpha * ket0 + beta * ket1;
            
            Console.WriteLine($"|psi> = {psi}, normalized? {IsNormalized(psi)}");

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
            // Pauli-X gate (quantum NOT / bit-flip):
            // X = [ 0  1 ]    X|0⟩ = |1⟩
            //     [ 1  0 ]    X|1⟩ = |0⟩
            var X = DenseMatrix.OfArray(new Complex[,] {
                { Complex.Zero, Complex.One },
                { Complex.One, Complex.Zero }
            });

            // Pauli-Z gate (phase-flip):
            // Z = [ 1   0 ]    Z|0⟩ = |0⟩
            //     [ 0  -1 ]    Z|1⟩ = -|1⟩
            var Z = DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero },
                { Complex.Zero, -Complex.One }
            });

            // Hadamard gate (superposition creator):
            // H = (1/√2) [ 1   1 ]    H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩
            //            [ 1  -1 ]    H|1⟩ = (|0⟩ - |1⟩)/√2 = |-⟩
            var H = (Complex) (1.0/Math.Sqrt(2)) * DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.One },
                { Complex.One, -Complex.One }
            });

            // Verify unitarity: U†U = I (unitary gates are reversible and preserve norm)
            Console.WriteLine($"Is X unitary? {IsUnitary(X)}");
            Console.WriteLine($"Is Z unitary? {IsUnitary(Z)}");
            Console.WriteLine($"Is H unitary? {IsUnitary(H)}");
        }

        /// <summary>
        /// Applies the Hadamard gate to create superposition.
        /// H|0⟩ = (1/√2)(|0⟩ + |1⟩) - equal superposition state |+⟩
        /// This is the most common way to create superposition in quantum algorithms.
        /// </summary>
        static void ApplyHadamardGate()
        {
            var ket0 = DenseVector.OfArray([Complex.One, Complex.Zero]);

            // Hadamard gate definition
            var H = (Complex) (1.0/Math.Sqrt(2)) * DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.One },
                { Complex.One, -Complex.One }
            });

            // Apply H to |0⟩: transforms basis state into superposition
            // H|0⟩ = (1/√2)[1, 1]ᵀ ≈ [0.707, 0.707]ᵀ
            var result = H * ket0;
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
            var ket0 = DenseVector.OfArray([Complex.One, Complex.Zero]);
            var ket1 = DenseVector.OfArray([Complex.Zero, Complex.One]);

            // Create 2-qubit computational basis states via tensor product
            // |00⟩ = |0⟩ ⊗ |0⟩ = [1,0,0,0]ᵀ
            var ket00 = Tensor(ket0, ket0);
            
            // |01⟩ = |0⟩ ⊗ |1⟩ = [0,1,0,0]ᵀ
            var ket01 = Tensor(ket0, ket1);
            
            // |10⟩ = |1⟩ ⊗ |0⟩ = [0,0,1,0]ᵀ
            var ket10 = Tensor(ket1, ket0);
            
            // |11⟩ = |1⟩ ⊗ |1⟩ = [0,0,0,1]ᵀ
            var ket11 = Tensor(ket1, ket1);

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
            // CNOT matrix with control qubit first, target qubit second:
            // [ 1  0  0  0 ]    |00⟩ → |00⟩ (control=0, do nothing)
            // [ 0  1  0  0 ]    |01⟩ → |01⟩ (control=0, do nothing)
            // [ 0  0  0  1 ]    |10⟩ → |11⟩ (control=1, flip target)
            // [ 0  0  1  0 ]    |11⟩ → |10⟩ (control=1, flip target)
            var CNOT = DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero, Complex.Zero, Complex.Zero },
                { Complex.Zero, Complex.One, Complex.Zero, Complex.Zero },
                { Complex.Zero, Complex.Zero, Complex.Zero, Complex.One },
                { Complex.Zero, Complex.Zero, Complex.One, Complex.Zero }
            });

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
            var ket0 = DenseVector.OfArray([Complex.One, Complex.Zero]);
            var ket00 = Tensor(ket0, ket0);

            // Hadamard gate for first qubit
            var H = (Complex) (1.0/Math.Sqrt(2)) * DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.One },
                { Complex.One, -Complex.One }
            });

            // Identity gate for second qubit (do nothing)
            var I2 = DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero },
                { Complex.Zero, Complex.One }
            });

            // CNOT gate
            var CNOT = DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero, Complex.Zero, Complex.Zero },
                { Complex.Zero, Complex.One, Complex.Zero, Complex.Zero },
                { Complex.Zero, Complex.Zero, Complex.Zero, Complex.One },
                { Complex.Zero, Complex.Zero, Complex.One, Complex.Zero }
            });

            // Step 1: Apply (H ⊗ I) to |00⟩ → creates superposition on first qubit
            // (H ⊗ I)|00⟩ = (|0⟩ + |1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
            var HtensorI = Kronecker(H, I2);
            var bellPre = HtensorI * ket00;

            // Step 2: Apply CNOT → creates entanglement
            // CNOT(|00⟩ + |10⟩)/√2 = (|00⟩ + |11⟩)/√2 = |Φ+⟩ (Bell state)
            var bell = CNOT * bellPre;

            Console.WriteLine($"Bell state (Phi+): {bell}, normalized? {IsNormalized(bell)}");
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
            // Pauli-Z gate (diagonal matrix - eigenvalues on diagonal)
            // Z = [ 1   0 ]
            //     [ 0  -1 ]
            var Z = DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero },
                { Complex.Zero, -Complex.One }
            });

            var ket0 = DenseVector.OfArray([Complex.One, Complex.Zero]);
            var ket1 = DenseVector.OfArray([Complex.Zero, Complex.One]);

            // Z|0⟩ = +1|0⟩ (eigenvalue +1)
            Console.WriteLine($"Z|0> = {Z * ket0}");
            
            // Z|1⟩ = -1|1⟩ (eigenvalue -1)
            Console.WriteLine($"Z|1> = {Z * ket1}");
            
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
            var ket0 = DenseVector.OfArray([Complex.One, Complex.Zero]);
            var ket1 = DenseVector.OfArray([Complex.Zero, Complex.One]);

            // Create equal superposition: |ψ⟩ = (|0⟩ + |1⟩)/√2
            // Each amplitude has magnitude 1/√2, so probability = (1/√2)² = 1/2
            var alpha = new Complex(1.0/Math.Sqrt(2), 0);
            var beta  = new Complex(1.0/Math.Sqrt(2), 0);
            var psi = alpha * ket0 + beta * ket1;

            // Calculate measurement probabilities in Z basis (computational basis)
            // P(0) = |⟨0|ψ⟩|² = |α|² = 1/2
            var p0 = ProbabilityOfOutcome(psi, ket0);
            
            // P(1) = |⟨1|ψ⟩|² = |β|² = 1/2
            var p1 = ProbabilityOfOutcome(psi, ket1);
            
            Console.WriteLine($"P_Z(0) = {p0:F4}, P_Z(1) = {p1:F4} (sum={p0+p1:F4})");
        }

        // Utilities

        static bool IsNormalized(Vector<Complex> vector)
        {
            double sum = 0.0;
            for (int i = 0; i < vector.Count; i++)
            {
                sum += Math.Pow(vector[i].Magnitude, 2);
            }
            return Math.Abs(sum - 1.0) < 1e-9;
        }

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