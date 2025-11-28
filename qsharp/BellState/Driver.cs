using System;
using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;

namespace BellState
{
    class Driver
    {
        static void Main(string[] args)
        {
            using var sim = new QuantumSimulator();
            var results = BellState.PrepareBellState.Run(sim).Result;
            Console.WriteLine($"Measurements: {results[0]}, {results[1]}");
            Console.WriteLine("Expect strong correlation (00 or 11).");
        }
    }
}