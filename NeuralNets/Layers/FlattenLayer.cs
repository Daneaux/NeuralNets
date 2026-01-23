using MatrixLibrary;
using System.Diagnostics;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    public class FlattenLayer : Layer
    {
        public FlattenLayer(InputOutputShape inputShape, int nodeCount, int randomSeed = 12341324) : base(inputShape, nodeCount, randomSeed)
        {
            int totSize = inputShape.TotalFlattenedSize;
            OutputShape = new InputOutputShape(1, totSize, 1, 1);
        }

        public override InputOutputShape OutputShape { get; }

        public override Tensor BackPropagation(Tensor dE_dY)
        {
            Debug.Assert(dE_dY.ToColumnVector() != null);
            ColumnVectorBase input = dE_dY.ToColumnVector();
            Debug.Assert(input.Size == InputShape.TotalFlattenedSize);

            List<MatrixBase> result = new List<MatrixBase>();
            // UN FLATTEN!
            int k = 0;
            for(int i = 0; i < InputShape.Count; i++) // not sure if it's Count or Depth ... 
            {
                MatrixBase mat = MatrixFactory.CreateMatrix(InputShape.Height, InputShape.Width);
                for (int r = 0; r < InputShape.Height; r++)
                    for (int c = 0; c < InputShape.Width; c++)
                        mat[r, c] = input[k++];

                result.Add(mat);
            }
            return result.ToTensor();
        }

        public override Tensor FeedFoward(Tensor input)
        {
            // flatten the ugliest way possible!
            if (input.ToColumnVector() != null) 
                return input;
            else 
            {
                var column = MatrixHelpers.UnrollMatricesToColumnVector(input.Matrices);
                return column.ToTensor();
            }
        }
    }
}
