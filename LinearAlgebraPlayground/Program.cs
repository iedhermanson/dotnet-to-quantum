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

            // Real vectors (double)
            var v = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray(new double[] { 1, 2, 3 });
            var w = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray(new double[] { 4, 5, 6 });

            var vPlusW = v + w;
            var scalarTimesV = 2.0 * v;
            var dot_vw = v.DotProduct(w);

            Console.WriteLine($"v + w = {vPlusW}");
            Console.WriteLine($"2 * v = {scalarTimesV}");
            Console.WriteLine($"dot(v, w) = {dot_vw}");

            // Complex vectors
            var vc = DenseVector.OfArray(new Complex[] { new Complex(1, 0), new Complex(0, 1) });
            var wc = DenseVector.OfArray(new Complex[] { new Complex(2, -1), new Complex(3, 4) });

            var inner_vc_wc = vc.ConjugateDotProduct(wc);
            Console.WriteLine($"<v|w> (complex inner product) = {inner_vc_wc}");

            // Real matrix
            var A = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(new double[,] {
                {1, 2},
                {3, 4}
            });
            var x = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray(new double[] {1, 0});
            Console.WriteLine($"A * x = {A * x}");
        }

        // Week 2: qubit states, normalization, probabilities
        static void Week2()
        {
            Console.WriteLine("\n--- Week 2: Quantum State Representation ---");

            // |0> = [1, 0]^T, |1> = [0, 1]^T (complex vectors)
            var ket0 = DenseVector.OfArray(new Complex[] { Complex.One, Complex.Zero });
            var ket1 = DenseVector.OfArray(new Complex[] { Complex.Zero, Complex.One });

            Console.WriteLine($"|0> = {ket0}");
            Console.WriteLine($"|1> = {ket1}");

            // Check normalization: sum |amplitude|^2 == 1
            Console.WriteLine($"Is |0> normalized? {IsNormalized(ket0)}");
            Console.WriteLine($"Is |1> normalized? {IsNormalized(ket1)}");

            // Create a superposition state alpha|0> + beta|1>, with normalization
            var alpha = new Complex(1.0/Math.Sqrt(2), 0);
            var beta  = new Complex(1.0/Math.Sqrt(2), 0);
            var psi = alpha * ket0 + beta * ket1;
            Console.WriteLine($"|psi> = {psi}, normalized? {IsNormalized(psi)}");

            // Probability of measuring 0 or 1 in computational basis
            var p0 = ProbabilityOfOutcome(psi, ket0);
            var p1 = ProbabilityOfOutcome(psi, ket1);
            Console.WriteLine($"P(0) = {p0:F4}, P(1) = {p1:F4} (sum={p0+p1:F4})");
        }

        // Week 3: gates as matrices, unitarity, apply gates
        static void Week3()
        {
            Console.WriteLine("\n--- Week 3: Quantum Gates as Matrices ---");

            var X = DenseMatrix.OfArray(new Complex[,] {
                { Complex.Zero, Complex.One },
                { Complex.One, Complex.Zero }
            });

            var Z = DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero },
                { Complex.Zero, -Complex.One }
            });

            var H = (Complex) (1.0/Math.Sqrt(2)) * DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.One },
                { Complex.One, -Complex.One }
            });

            Console.WriteLine($"Is X unitary? {IsUnitary(X)}");
            Console.WriteLine($"Is Z unitary? {IsUnitary(Z)}");
            Console.WriteLine($"Is H unitary? {IsUnitary(H)}");

            var ket0 = DenseVector.OfArray(new Complex[] { Complex.One, Complex.Zero });
            var ket1 = DenseVector.OfArray(new Complex[] { Complex.Zero, Complex.One });

            var result = H * ket0; // Hadamard creates superposition
            Console.WriteLine($"H|0> = {result}");
        }

        // Week 4: tensor products, 2-qubit states, CNOT
        static void Week4()
        {
            Console.WriteLine("\n--- Week 4: Multi-Qubit Systems ---");

            var ket0 = DenseVector.OfArray(new Complex[] { Complex.One, Complex.Zero });
            var ket1 = DenseVector.OfArray(new Complex[] { Complex.Zero, Complex.One });

            // |00> = |0> ⊗ |0>
            var ket00 = Tensor(ket0, ket0);
            // |01> = |0> ⊗ |1>
            var ket01 = Tensor(ket0, ket1);
            // |10> = |1> ⊗ |0>
            var ket10 = Tensor(ket1, ket0);
            // |11> = |1> ⊗ |1>
            var ket11 = Tensor(ket1, ket1);

            Console.WriteLine($"|00> = {ket00}");
            Console.WriteLine($"|01> = {ket01}");

            // CNOT matrix (control qubit first)
            var CNOT = DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero, Complex.Zero, Complex.Zero }, // |00> -> |00>
                { Complex.Zero, Complex.One, Complex.Zero, Complex.Zero }, // |01> -> |01>
                { Complex.Zero, Complex.Zero, Complex.Zero, Complex.One }, // |10> -> |11>
                { Complex.Zero, Complex.Zero, Complex.One, Complex.Zero }  // |11> -> |10>
            });

            // Create (H ⊗ I)|00> then apply CNOT to make a Bell state
            var H = (Complex) (1.0/Math.Sqrt(2)) * DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.One },
                { Complex.One, -Complex.One }
            });

            var I2 = DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero },
                { Complex.Zero, Complex.One }
            });

            var HtensorI = Kronecker(H, I2);
            var bellPre = HtensorI * ket00;
            var bell = CNOT * bellPre;

            Console.WriteLine($"Bell state (Phi+): {bell}, normalized? {IsNormalized(bell)}");
        }

        // Week 5: measurement probabilities, eigenvalues/eigenvectors of Z
        static void Week5()
        {
            Console.WriteLine("\n--- Week 5: Measurement & Eigen Concepts ---");

            var Z = DenseMatrix.OfArray(new Complex[,] {
                { Complex.One, Complex.Zero },
                { Complex.Zero, -Complex.One }
            });

            // Eigen decomposition of Z (manual since it's simple)
            // Eigenvalues: +1 with eigenvector |0>, -1 with eigenvector |1>
            var ket0 = DenseVector.OfArray(new Complex[] { Complex.One, Complex.Zero });
            var ket1 = DenseVector.OfArray(new Complex[] { Complex.Zero, Complex.One });

            Console.WriteLine($"Z|0> = {Z * ket0}");
            Console.WriteLine($"Z|1> = {Z * ket1}");
            Console.WriteLine($"Eigenvalues (expected): +1 for |0>, -1 for |1>");

            // Measurement probabilities for an arbitrary 1-qubit state
            var alpha = new Complex(1.0/Math.Sqrt(2), 0);
            var beta  = new Complex(1.0/Math.Sqrt(2), 0);
            var psi = alpha * ket0 + beta * ket1;

            var p0 = ProbabilityOfOutcome(psi, ket0);
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