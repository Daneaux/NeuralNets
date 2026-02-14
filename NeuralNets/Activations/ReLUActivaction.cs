using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System.Diagnostics;
using System.Net.NetworkInformation;

namespace NeuralNets
{

    public class ReLUActivaction : Layer, IActivationFunction
    {
        public ReLUActivaction() : base(new InputOutputShape(1,1,1,1), 1)
        {
        }

        public Tensor LastActivation
        {
            get; private set;
        }

        public override InputOutputShape OutputShape => throw new NotImplementedException();

        public override Tensor FeedFoward(Tensor input)
        {
            if (input.ToColumnVector() != null)
                return Activate(input.ToColumnVector()).ToTensor();
            else
                return Activate(input.Matrices).ToTensor();
        }
        public override Tensor BackPropagation(Tensor dE_dX)
        {
            bool debugMode = Environment.GetEnvironmentVariable("NEURALNET_DEBUG") == "1";
            
            if (debugMode)
            {
                Console.WriteLine($"\n  [ReLU.BackPropagation] START");
                if (LastActivation?.ToColumnVector() != null)
                {
                    Console.WriteLine($"    Stored LastActivation (from forward): [{string.Join(", ", Enumerable.Range(0, LastActivation.ToColumnVector().Size).Select(i => LastActivation.ToColumnVector()[i].ToString("F6")))}]");
                }
                if (dE_dX.ToColumnVector() != null)
                {
                    Console.WriteLine($"    Incoming dE/dX: [{string.Join(", ", Enumerable.Range(0, dE_dX.ToColumnVector().Size).Select(i => dE_dX.ToColumnVector()[i].ToString("F6")))}]");
                }
            }

            // ReLU derivative: multiply incoming gradient by derivative of LastActivation
            // If LastActivation[i] > 0: derivative is 1, so pass gradient through
            // If LastActivation[i] = 0: derivative is 0, so block gradient
            if(dE_dX.IsVector)
            {
                var derivative = this.Derivative(this.LastActivation.ToColumnVector());
                
                if (debugMode)
                {
                    Console.WriteLine($"    ReLU derivative mask (>0? 1 : 0): [{string.Join(", ", Enumerable.Range(0, derivative.Size).Select(i => derivative[i].ToString()))}]");
                }
                
                var result = derivative * dE_dX.ToColumnVector();
                
                if (debugMode)
                {
                    Console.WriteLine($"    Output (dE/dX * derivative): [{string.Join(", ", Enumerable.Range(0, result.Size).Select(i => result[i].ToString("F6")))}]");
                    Console.WriteLine($"  [ReLU.BackPropagation] END");
                }
                
                return result.ToTensor();
            }
            else
            {
                Debug.Assert(dE_dX.IsMatrix && !dE_dX.IsVector); 
                var derivative = this.Derivative(this.LastActivation.Matrices);
                // Multiply element-wise
                var dE_dX_mats = dE_dX.Matrices;
                var result = new List<MatrixBase>();
                for(int i = 0; i < derivative.Count; i++)
                {
                    result.Add(derivative[i].Multiply(dE_dX_mats[i]));
                }
                return result.ToTensor();
            }
        }

        public ColumnVectorBase Activate(ColumnVectorBase input)
        {
            float[] floats = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
            {
                floats[i] = Math.Max(0, input[i]);
            }
            var activation = MatrixFactory.CreateColumnVector(floats);
            LastActivation = activation.ToTensor();
            return activation;
        }

        public MatrixBase Activate(MatrixBase input)
        {
            return Activate(new List<MatrixBase> { input })[0];
        }

        public List<MatrixBase> Activate(List<MatrixBase> input)
        {
            int rows = input[0].Rows;
            int cols = input[0].Cols;
            List<MatrixBase> resultList = new List<MatrixBase>();
            foreach (MatrixBase mat in input)
            {
                MatrixBase result = MatrixFactory.CreateMatrix(rows, cols);
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        result[r, c] = Math.Max(0, mat[r, c]);
                    }
                }
                resultList.Add(result);
            }
            LastActivation = resultList.ToTensor();
            return resultList;
        }

        public ColumnVectorBase Derivative(ColumnVectorBase lastActivation)
        {
            if (lastActivation == null)
                throw new InvalidOperationException();

            float[] derivative = new float[lastActivation.Size];
            for (int i = 0; i < lastActivation.Size; i++)
            {
                // ReLU derivative: 1 if output > 0, 0 if output <= 0
                // Note: ReLU output is always >= 0, so we check if > 0
                if (lastActivation[i] > 0)
                    derivative[i] = 1;
                else
                    derivative[i] = 0;
            }
            ColumnVectorBase dVec =  MatrixFactory.CreateColumnVector(derivative);
            return dVec;
        }
        public List<MatrixBase> Derivative(List<MatrixBase> lastActivation)
        {
            if (lastActivation == null)
                throw new InvalidOperationException();

            List<MatrixBase> result = new List<MatrixBase>();
            foreach(MatrixBase activationMat in lastActivation)
            {
                var dMat = MatrixFactory.CreateMatrix(activationMat.Rows, activationMat.Cols);
                for(int r = 0; r < activationMat.Rows; r++)
                    for(int c = 0; c < activationMat.Cols; c++)
                    {
                        // ReLU derivative: 1 if output > 0, 0 if output <= 0
                        if (activationMat[r, c] > 0)
                            dMat[r, c] = 1;
                        else 
                            dMat[r, c] = 0;
                    }
                result.Add(dMat);
            }

            return result;
        }
    }

    public class ReLUActivaction_old : IActivationFunction
    {
        public Tensor LastActivation => throw new NotImplementedException();

        public ColumnVectorBase Activate(ColumnVectorBase input)
        {
            float[] floats = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
            {
                floats[i] = Math.Max(0, input[i]);
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
                    result[r, c] = Math.Max(0, input[r, c]);
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

            float[] derivative = new float[lastActivation.Size];
            for (int i = 0; i < lastActivation.Size; i++)
            {
                if (lastActivation[i] >= 0)
                    derivative[i] = 1;
                else
                    derivative[i] = 0;
            }
            ColumnVectorBase dVec = MatrixFactory.CreateColumnVector(derivative);
            return dVec;
        }
    }
}