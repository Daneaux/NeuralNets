using MatrixLibrary;
using System.Runtime.CompilerServices;
using TorchSharp;

namespace NeuralNets
{
    public interface IActivationFunction
    {
        ColumnVector Activate(ColumnVector input);
        AvxColumnVector Activate(AvxColumnVector input);
        ColumnVector Derivative(ColumnVector lastActivation);
        AvxColumnVector Derivative(AvxColumnVector lastActivation);
        AvxMatrix Activate(AvxMatrix input);
    }

    public class ReLUActivaction : IActivationFunction
    {
        public ColumnVector Activate(ColumnVector input)
        {
            ColumnVector activation = new ColumnVector(input.Size);
            for (int i = 0; i < input.Size; i++)
            {
                activation[i] = Math.Max(0, input[i]);
            }
            return activation;
        }

        public AvxColumnVector Activate(AvxColumnVector input)
        {
            float[] floats = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
            {
                floats[i] = Math.Max(0, input[i]);
            }
            return new AvxColumnVector(floats);
        }

        public ColumnVector Derivative(ColumnVector lastActivation)
        {
            if (lastActivation == null)
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

        public AvxMatrix Activate(AvxMatrix input)
        {
            AvxMatrix result = new AvxMatrix(input.Rows, input.Cols);
            for (int r = 0; r < input.Rows; r++)
            {
                for (int c = 0; c < input.Cols; c++)
                {
                    result[r, c] = Math.Max(0, input[r, c]);
                }
            }
            return result;
        }

        public AvxColumnVector Derivative(AvxColumnVector lastActivation)
        {
            if (lastActivation == null)
            {
                throw new InvalidOperationException();
            }

            float[] derivative = new float[lastActivation.Size];
            for (int i = 0; i < lastActivation.Size; i++)
            {
                if (lastActivation[i] >= 0)
                    derivative[i] = 1;
                else
                    derivative[i] = 0;
            }
            AvxColumnVector dVec = new AvxColumnVector(derivative);
            return dVec;
        }
    }

   public class SigmoidActivation : IActivationFunction
    {
        public ColumnVector Activate(ColumnVector input)
        {
            ColumnVector activation = new ColumnVector(input.Size);
            for (int i = 0; i < input.Size; i++)
            {
                activation[i] = MathPowE(input[i]);
            }
            return activation;
        }

        public AvxColumnVector Activate(AvxColumnVector input)
        {
            float[] floats = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
            {
                floats[i] = MathPowE(input[i]);
            }
            return new AvxColumnVector(floats);
        }

        public AvxMatrix Activate(AvxMatrix input)
        {
            AvxMatrix result = new AvxMatrix(input.Rows, input.Cols);
            for (int r = 0; r < input.Rows; r++)
            {
                for (int c = 0; c < input.Cols; c++)
                {
                    result[r, c] = MathPowE(input[r, c]);
                }
            }
            return result;
        }

        public ColumnVector Derivative(ColumnVector lastActivation)
        {
            if (lastActivation == null)
            {
                throw new InvalidOperationException();
            }

            ColumnVector derivative = lastActivation * (1 - lastActivation);
            return derivative;
        }

        public AvxColumnVector Derivative(AvxColumnVector lastActivation)
        {
            if (lastActivation == null)
            {
                throw new InvalidOperationException();
            }

            AvxColumnVector derivative = lastActivation * (1f - lastActivation);
            return derivative;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float MathPowE(float x)
        {
            return 1.0f / (1.0f + (float)Math.Pow(Math.E, -x));
        }
    }

}