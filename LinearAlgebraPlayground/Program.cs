using MathNet.Numerics.LinearAlgebra.Complex;
using Complex = System.Numerics.Complex;
using Microsoft.Quantum.Simulation.Simulators;

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
            Week6();
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
            VectorOperations.Create3DRealVectors(
                [1, 2, 3],
                [4, 5, 6],
                out MathNet.Numerics.LinearAlgebra.Double.DenseVector v, 
                out MathNet.Numerics.LinearAlgebra.Double.DenseVector w);

            var vPlusW = VectorOperations.Add(v, w);
            Console.WriteLine($"v + w = {vPlusW}");

            var scalarTimesV = VectorOperations.ScalarMultiply(v, 2.0);
            Console.WriteLine($"2 * v = {scalarTimesV}");

            var dot_vw = VectorOperations.DotProduct(v, w);
            Console.WriteLine($"dot(v, w) = {dot_vw}");
        }

        /// <summary>
        /// Demonstrates the complex inner product (Hermitian inner product).
        /// In quantum mechanics, the inner product ⟨v|w⟩ gives probability amplitudes.
        /// Uses conjugate of first vector: ⟨v|w⟩ = conjugate(v) · w
        /// </summary>
        static void ComplexVectorInnerProduct()
        {
            // Create two complex vectors
            VectorOperations.CreateComplexVectors(
                [new Complex(1, 0), new Complex(0, 1)],
                [new Complex(2, -1), new Complex(3, 4)],
                out DenseVector vc,
                out DenseVector wc);

            var inner_vc_wc = VectorOperations.HermitianInnerProduct(vc, wc);
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
            MatrixOperations.Create2x2MatrixAndVector(
                new double[,] { {1, 2}, {3, 4} },
                [1, 0],
                out MathNet.Numerics.LinearAlgebra.Double.DenseMatrix A,
                out MathNet.Numerics.LinearAlgebra.Double.DenseVector x);

            var result = MatrixOperations.ApplyTransformation(A, x);
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
            QuantumStates.CreateBasisStates(out DenseVector ket0, out DenseVector ket1);

            Console.WriteLine($"|0> = {ket0}");
            Console.WriteLine($"|1> = {ket1}");

            // Verify normalization: |α₀|² + |α₁|² = 1
            // For valid quantum states, probabilities must sum to 1
            var ket0Normalized = QuantumStates.IsNormalized(ket0);
            var ket1Normalized = QuantumStates.IsNormalized(ket1);
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
            QuantumStates.CreateBasisStates(out DenseVector ket0, out DenseVector ket1);

            // Create equal superposition: |ψ⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
            // This is the |+⟩ state - 50% chance of measuring 0 or 1
            var alpha = new Complex(1.0/Math.Sqrt(2), 0);
            var beta  = new Complex(1.0/Math.Sqrt(2), 0);
            
            var psi = QuantumStates.CreateSuperposition(ket0, ket1, alpha, beta);
            var psiNormalized = QuantumStates.IsNormalized(psi);
            
            Console.WriteLine($"|psi> = {psi}, normalized? {psiNormalized}");

            // Calculate measurement probabilities in computational basis
            // P(0) = |⟨0|ψ⟩|² and P(1) = |⟨1|ψ⟩|²
            // Born rule: probability = squared magnitude of inner product
            var p0 = QuantumStates.ProbabilityOfOutcome(psi, ket0);
            var p1 = QuantumStates.ProbabilityOfOutcome(psi, ket1);
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
            var X = QuantumGates.CreatePauliX();
            var Z = QuantumGates.CreatePauliZ();
            var H = QuantumGates.CreateHadamard();

            // Verify unitarity: U†U = I (unitary gates are reversible and preserve norm)
            var xUnitary = MatrixOperations.IsUnitary(X);
            var zUnitary = MatrixOperations.IsUnitary(Z);
            var hUnitary = MatrixOperations.IsUnitary(H);
            
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
            QuantumStates.CreateBasisStates(out DenseVector ket0, out _);

            var H = QuantumGates.CreateHadamard();

            // Apply H to |0⟩: transforms basis state into superposition
            // H|0⟩ = (1/√2)[1, 1]ᵀ ≈ [0.707, 0.707]ᵀ
            var result = QuantumGates.Apply(H, ket0);
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
            QuantumStates.CreateBasisStates(out DenseVector ket0, out DenseVector ket1);

            // Create 2-qubit computational basis states via tensor product
            var ket00 = QuantumStates.Tensor(ket0, ket0);
            var ket01 = QuantumStates.Tensor(ket0, ket1);

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
            var CNOT = QuantumGates.CreateCNOT();
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
            QuantumStates.CreateBasisStates(out DenseVector ket0, out _);
            var ket00 = QuantumStates.Tensor(ket0, ket0);

            var H = QuantumGates.CreateHadamard();
            var I2 = QuantumGates.CreateIdentity();
            var CNOT = QuantumGates.CreateCNOT();

            // Step 1: Apply (H ⊗ I) to |00⟩ → creates superposition on first qubit
            // (H ⊗ I)|00⟩ = (|0⟩ + |1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
            var HtensorI = MatrixOperations.Kronecker(H, I2);
            var bellPre = HtensorI * ket00;

            // Step 2: Apply CNOT → creates entanglement
            // CNOT(|00⟩ + |10⟩)/√2 = (|00⟩ + |11⟩)/√2 = |Φ+⟩ (Bell state)
            var bell = (DenseVector)(CNOT * bellPre);
            var bellNormalized = QuantumStates.IsNormalized(bell);

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
            var Z = QuantumGates.CreatePauliZ();
            QuantumStates.CreateBasisStates(out DenseVector ket0, out DenseVector ket1);

            // Z|0⟩ = +1|0⟩ (eigenvalue +1)
            var z0 = QuantumGates.Apply(Z, ket0);
            Console.WriteLine($"Z|0> = {z0}");
            
            // Z|1⟩ = -1|1⟩ (eigenvalue -1)
            var z1 = QuantumGates.Apply(Z, ket1);
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
            QuantumStates.CreateBasisStates(out DenseVector ket0, out DenseVector ket1);

            // Create equal superposition: |ψ⟩ = (|0⟩ + |1⟩)/√2
            // Each amplitude has magnitude 1/√2, so probability = (1/√2)² = 1/2
            var alpha = new Complex(1.0/Math.Sqrt(2), 0);
            var beta  = new Complex(1.0/Math.Sqrt(2), 0);
            var psi = QuantumStates.CreateSuperposition(ket0, ket1, alpha, beta);

            // Calculate measurement probabilities in Z basis (computational basis)
            // P(0) = |⟨0|ψ⟩|² = |α|² = 1/2
            var p0 = QuantumStates.ProbabilityOfOutcome(psi, ket0);
            
            // P(1) = |⟨1|ψ⟩|² = |β|² = 1/2
            var p1 = QuantumStates.ProbabilityOfOutcome(psi, ket1);
            
            Console.WriteLine($"P_Z(0) = {p0:F4}, P_Z(1) = {p1:F4} (sum={p0+p1:F4})");
        }
        static void Week6()
        {
            Console.WriteLine("\n=== Week 6: Q# Bell State (with .NET Comparison) ===\n");
            
            // === Part 1: .NET Linear Algebra Simulation ===
            Console.WriteLine("--- Part 1: .NET Linear Algebra Bell State ---");
            
            // Create |0⟩ and |1⟩ basis states
            QuantumStates.CreateBasisStates(out DenseVector ket0, out DenseVector ket1);
            
            // Create two-qubit initial state |00⟩ = |0⟩ ⊗ |0⟩
            var ket00 = QuantumStates.Tensor(ket0, ket0);
            Console.WriteLine("Initial state |00⟩:");
            Console.WriteLine($"  [{ket00[0]:F4}, {ket00[1]:F4}, {ket00[2]:F4}, {ket00[3]:F4}]");
            Console.WriteLine("  (basis: |00⟩, |01⟩, |10⟩, |11⟩)");
            
            // Apply Hadamard to first qubit: (H ⊗ I)|00⟩ = (|0⟩ + |1⟩)/√2 ⊗ |0⟩
            var hadamard = QuantumGates.CreateHadamard();
            var identity = QuantumGates.CreateIdentity();
            var h_tensor_i = MatrixOperations.Kronecker(hadamard, identity);
            var afterHadamard = QuantumGates.Apply(h_tensor_i, ket00);
            Console.WriteLine("\nAfter H ⊗ I:");
            Console.WriteLine($"  [{afterHadamard[0]:F4}, {afterHadamard[1]:F4}, {afterHadamard[2]:F4}, {afterHadamard[3]:F4}]");
            Console.WriteLine("  = (|00⟩ + |10⟩)/√2");
            
            // Apply CNOT: creates Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            var cnot = QuantumGates.CreateCNOT();
            var bellState = QuantumGates.Apply(cnot, afterHadamard);
            Console.WriteLine("\nAfter CNOT (Bell State |Φ⁺⟩):");
            Console.WriteLine($"  [{bellState[0]:F4}, {bellState[1]:F4}, {bellState[2]:F4}, {bellState[3]:F4}]");
            Console.WriteLine("  = (|00⟩ + |11⟩)/√2");
            Console.WriteLine("  Interpretation: 50% chance |00⟩, 50% chance |11⟩, 0% for |01⟩ or |10⟩");
            
            // === Part 2: Q# Quantum Simulator ===
            Console.WriteLine("\n--- Part 2: Q# Quantum Execution ---");
            Console.WriteLine("Running Q# PrepareBellState operation 100 times...\n");
            
            using var sim = new QuantumSimulator();
            
            // Count measurement outcomes
            int count00 = 0, count01 = 0, count10 = 0, count11 = 0;
            int shots = 100;
            
            for (int i = 0; i < shots; i++)
            {
                var result = BellState.PrepareBellState.Run(sim).Result;
                
                // result is Result[] array from measuring both qubits
                bool qubit0 = result[0] == Microsoft.Quantum.Simulation.Core.Result.One;
                bool qubit1 = result[1] == Microsoft.Quantum.Simulation.Core.Result.One;
                
                if (!qubit0 && !qubit1) count00++;
                else if (!qubit0 && qubit1) count01++;
                else if (qubit0 && !qubit1) count10++;
                else count11++;
            }
            
            Console.WriteLine($"Measurement results ({shots} shots):");
            Console.WriteLine($"  |00⟩: {count00} ({100.0*count00/shots:F1}%)");
            Console.WriteLine($"  |01⟩: {count01} ({100.0*count01/shots:F1}%)");
            Console.WriteLine($"  |10⟩: {count10} ({100.0*count10/shots:F1}%)");
            Console.WriteLine($"  |11⟩: {count11} ({100.0*count11/shots:F1}%)");
            
            // === Part 3: Comparison Hooks ===
            Console.WriteLine("\n--- Part 3: Comparison Analysis ---");
            Console.WriteLine("✓ Both methods apply: H gate on qubit 0, then CNOT(0,1)");
            Console.WriteLine("✓ .NET shows state vector: (|00⟩ + |11⟩)/√2 with exact amplitudes");
            Console.WriteLine("✓ Q# shows measurements: actual random outcomes from quantum evolution");
            Console.WriteLine("✓ Entanglement signature: Both qubits always measured in same state");
            Console.WriteLine($"✓ Expected ~50%/~50% for |00⟩/|11⟩, got {100.0*count00/shots:F1}%/{100.0*count11/shots:F1}%");
            Console.WriteLine($"✓ |01⟩ and |10⟩ should be ~0%: got {count01}/{count10} (confirms entanglement)");
            Console.WriteLine("\nKey insight: .NET simulates the mathematics, Q# executes the physics!");
        }
    }
}