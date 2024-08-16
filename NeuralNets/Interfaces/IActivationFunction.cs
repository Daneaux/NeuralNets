using System;
using System.Diagnostics;

namespace NeuralNets
{
    public interface IActivationFunction
    {
        ColumnVector? LastActivation { get; }
       // double Activate(double input);
        ColumnVector Activate(ColumnVector input);
        ColumnVector Derivative();
    }

    public class ReLUActivaction : IActivationFunction
    {
        // seems inconsistent, should last actiavtion belong to the layer?  BUG TODO
        public ColumnVector? LastActivation { get; private set; }

        public ColumnVector Activate(ColumnVector input)
        {
            LastActivation = new ColumnVector(input.Size);
            for (int i = 0; i < input.Size; i++)
            {
                LastActivation[i] = Math.Max(0, input[i]);
            }
            return LastActivation;
        }

        public ColumnVector Derivative()
        {
            if(LastActivation == null)
            {
                throw new InvalidOperationException();
            }

            ColumnVector derivative = new ColumnVector(LastActivation.Size);
            for (int i = 0; i < LastActivation.Size; i++)
            {
                if (LastActivation[i] >= 0)
                    derivative[i] = 1;
                else
                    derivative[i] = 0;
            }
            return derivative;
        }
    }

   public class SigmoidActivation : IActivationFunction
    {
        public ColumnVector? LastActivation { get; private set; }

        public double Activate(double input)
        {
            return (1.0 / (1.0 + Math.Pow(Math.E, -input)));
        }

        public ColumnVector Activate(ColumnVector input)
        {
            ColumnVector act = new ColumnVector(input.Size);
            for (int i = 0; i < input.Size; i++)
            {
                act[i] = (1.0 / (1.0 + Math.Pow(Math.E, -input[i])));
            }
            LastActivation = act;
            return act;
        }

        public ColumnVector Derivative()
        {
            Debug.Assert(LastActivation != null);
            ColumnVector derivative = this.LastActivation * (1.0 - this.LastActivation);
            //this.LastActivation = derivative; // bug?
            return derivative;
        }
    }

}