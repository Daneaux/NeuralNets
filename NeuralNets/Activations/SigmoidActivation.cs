using System.Runtime.CompilerServices;
using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using TorchSharp;

namespace NeuralNets
{
    public class SigmoidActivation_Old : IActivationFunction
    {
        public Tensor LastActivation => throw new NotImplementedException();

        public ColumnVectorBase Activate(ColumnVectorBase input)
        {
            float[] floats = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
            {
                floats[i] = MathPowE(input[i]);
            }
            return MatrixFactory.CreateColumnVector(floats);
        }

        public MatrixBase Activate(MatrixBase input)
        {
            MatrixBase result = MatrixFactory.CreateMatrix(input.Rows, input.Cols);
            for (int r = 0; r < input.Rows; r++)
            {
                for (int c = 0; c < input.Cols; c++)
                {
                    result[r, c] = MathPowE(input[r, c]);
                }
            }
            return result;
        }

        public List<MatrixBase> Activate(List<MatrixBase> input)
        {
            throw new NotImplementedException();
        }

        public ColumnVectorBase Derivative(ColumnVectorBase lastActivation)
        {
            if (lastActivation == null)
            {
                throw new InvalidOperationException();
            }

            ColumnVectorBase derivative = lastActivation * (1f - lastActivation);
            return derivative;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float MathPowE(float x)
        {
            return 1.0f / (1.0f + (float)Math.Pow(Math.E, -x));
        }
    }
    public class SigmoidActivation : Layer, IActivationFunction   
    {
        public SigmoidActivation() : base(new InputOutputShape(1,1,1,1), 1)
        {
        }

        public override InputOutputShape OutputShape => new InputOutputShape(1, NumNodes, 1, 1);

        public override Tensor FeedFoward(Tensor input)
        {
            if(input.ToColumnVector() != null)
            {
                return this.Activate(input.ToColumnVector()).ToTensor();
            } 
            else if(input.Matrices != null)
            {
                return this.Activate(input.Matrices).ToTensor();
            }
            else 
            {
                throw new InvalidOperationException();
            }
        }

        public override Tensor BackPropagation(Tensor dE_dX)
        {
            //return Derivative(dE_dX.ToColumnVector()).ToTensor();

            bool debugMode = Environment.GetEnvironmentVariable("NEURALNET_DEBUG") == "1";

            if (dE_dX.IsVector)
            {
                var derivative = this.Derivative(this.LastActivation.ToColumnVector());

                if (debugMode)
                {
                    Console.WriteLine($"    Sigmoid derivative mask (>0? 1 : 0): [{string.Join(", ", Enumerable.Range(0, derivative.Size).Select(i => derivative[i].ToString()))}]");
                }

                var result = derivative * dE_dX.ToColumnVector();

                if (debugMode)
                {
                    Console.WriteLine($"    Output (dE/dX * derivative): [{string.Join(", ", Enumerable.Range(0, result.Size).Select(i => result[i].ToString("F6")))}]");
                    Console.WriteLine($"  [Sigmoig.BackPropagation] END");
                }

                return result.ToTensor();
            }
            else
            {
                throw new NotImplementedException("Not expeecting matrix tensor in Sigmoid BackPropagation");
            }

        }

        public Tensor LastActivation { get; private set; }
        public ColumnVectorBase Activate(ColumnVectorBase input)
        {
            float[] floats = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
            {
                floats[i] = MathPowE(input[i]);
            }
            var result = MatrixFactory.CreateColumnVector(floats);
            LastActivation = result.ToTensor();
            return result;
        }

        public MatrixBase Activate(MatrixBase input)
        {
            MatrixBase result = MatrixFactory.CreateMatrix(input.Rows, input.Cols);
            for (int r = 0; r < input.Rows; r++)
            {
                for (int c = 0; c < input.Cols; c++)
                {
                    result[r, c] = MathPowE(input[r, c]);
                }
            }
            LastActivation = result.ToTensor();
            return result;
        }

        public List<MatrixBase> Activate(List<MatrixBase> input)
        {
            List<MatrixBase> resultList = new List<MatrixBase>();

            foreach (MatrixBase mat in input)
                resultList.Add(this.Activate(mat));

            LastActivation = resultList.ToTensor();
            return resultList;
        }

        public ColumnVectorBase Derivative(ColumnVectorBase lastActivation)
        {
            if (lastActivation == null)
            {
                throw new InvalidOperationException();
            }

            ColumnVectorBase derivative = lastActivation * (1f - lastActivation);
            return derivative;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float MathPowE(float x)
        {
            return 1.0f / (1.0f + (float)Math.Pow(Math.E, -x));
        }
    }

}