using System;
using System.Diagnostics;

namespace NeuralNets
{
    public interface IActivationFunction
    {
       // ColumnVector? LastActivation { get; }
        ColumnVector Activate(ColumnVector input);
        ColumnVector Derivative(ColumnVector lastActivation);
    }

    public class ReLUActivaction : IActivationFunction
    {
        // seems inconsistent, should last actiavtion belong to the layer?  BUG TODO
        //public ColumnVector? LastActivation { get; private set; }

        public ColumnVector Activate(ColumnVector input)
        {
            ColumnVector activation = new ColumnVector(input.Size);
            for (int i = 0; i < input.Size; i++)
            {
                activation[i] = Math.Max(0, input[i]);
            }
            return activation;
        }

        public ColumnVector Derivative(ColumnVector lastActivation)
        {
            if(lastActivation == null)
            {
                throw new InvalidOperationException();
            }

            ColumnVector derivative = new ColumnVector(lastActivation.Size);
            for (int i = 0; i < lastActivation.Size; i++)
            {
                if (lastActivation[i] >= 0)
                    derivative[i] = 1;
                else
                    derivative[i] = 0;
            }
            return derivative;
        }
    }

   public class SigmoidActivation : IActivationFunction
    {
        public ColumnVector Activate(ColumnVector input)
        {
            ColumnVector LastActivation = new ColumnVector(input.Size);
            for (int i = 0; i < input.Size; i++)
            {
                LastActivation[i] = (1.0 / (1.0 + Math.Pow(Math.E, -input[i])));
            }
            return LastActivation;
        }

        public ColumnVector Derivative(ColumnVector lastActivation)
        {
            if (lastActivation == null)
            {
                throw new InvalidOperationException();
            }

            ColumnVector derivative = lastActivation * (1.0 - lastActivation);
            return derivative;
        }
    }

}