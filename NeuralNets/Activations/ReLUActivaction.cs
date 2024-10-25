using MatrixLibrary;
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
            if (input.ToAvxColumnVector() != null)
                return Activate(input.ToAvxColumnVector()).ToTensor();
            else
                return Activate(input.Matrices).ToTensor();
        }
        public override Tensor BackPropagation(Tensor dE_dX)
        {
            if(dE_dX.ToAvxColumnVector() != null)
                return this.Derivative(dE_dX.ToAvxColumnVector()).ToTensor();
            else
                return this.Derivative(dE_dX.Matrices).ToTensor();
        }

        public AvxColumnVector Activate(AvxColumnVector input)
        {
            float[] floats = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
            {
                floats[i] = Math.Max(0, input[i]);
            }
            var activation = new AvxColumnVector(floats);
            LastActivation = activation.ToTensor();
            return activation;
        }

        public AvxMatrix Activate(AvxMatrix input)
        {
            return Activate(new List<AvxMatrix> { input })[0];
        }

        public List<AvxMatrix> Activate(List<AvxMatrix> input)
        {
            int rows = input[0].Rows;
            int cols = input[0].Cols;
            List<AvxMatrix> resultList = new List<AvxMatrix>();
            foreach (AvxMatrix mat in input)
            {
                AvxMatrix result = new AvxMatrix(rows, cols);
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

        public AvxColumnVector Derivative(AvxColumnVector lastActivation)
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
            AvxColumnVector dVec = new AvxColumnVector(derivative);
            return dVec;
        }
        public List<AvxMatrix> Derivative(List<AvxMatrix> lastActivation)
        {
            if (lastActivation == null)
                throw new InvalidOperationException();

            List<AvxMatrix> result = new List<AvxMatrix>();
            foreach(AvxMatrix activationMat in lastActivation)
            {
                var dMat = new AvxMatrix(activationMat.Rows, activationMat.Cols);
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

        public AvxColumnVector Activate(AvxColumnVector input)
        {
            float[] floats = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
            {
                floats[i] = Math.Max(0, input[i]);
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
                    result[r, c] = Math.Max(0, input[r, c]);
                }
            }
            return result;
        }

        public List<AvxMatrix> Activate(List<AvxMatrix> input)
        {
            throw new NotImplementedException();
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
}