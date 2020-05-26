using System;

namespace NeuralNets
{
    public interface IActivationFunction
    {
        double Activate(double weightedSum);
    }

    class ReLUActivaction : IActivationFunction
    {
        public double Activate(double input)
        {
            return Math.Max(0, input);
        }
    }

    class SigmoidActivation : IActivationFunction
    {
        public double Activate(double value)
        {
            return (1.0 / (1.0 + Math.Pow(Math.E, -value)));
        }
    }

}