using MatrixLibrary;
using MatrixLibrary.BaseClasses;
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
            if(dE_dX.ToColumnVector() != null)
                return this.Derivative(dE_dX.ToColumnVector()).ToTensor();
            else
                return this.Derivative(dE_dX.Matrices).ToTensor();
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
                if (lastActivation[i] >= 0)
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
                        if (activationMat[r, c] >= 0)
                            dMat[r, c] = 1;
                        else dMat[r, c] = 0;
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