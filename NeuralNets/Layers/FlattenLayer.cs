using MatrixLibrary;
using System.Diagnostics;

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
            Debug.Assert(dE_dY.ToAvxColumnVector() != null);
            AvxColumnVector input = dE_dY.ToAvxColumnVector();
            Debug.Assert(input.Size == InputShape.TotalFlattenedSize);

            List<AvxMatrix> result = new List<AvxMatrix>();
            // UN FLATTEN!
            int k = 0;
            for(int i = 0; i < InputShape.Count; i++) // not sure if it's Count or Depth ... 
            {
                AvxMatrix mat = new AvxMatrix(InputShape.Height, InputShape.Width);
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
            if (input.ToAvxColumnVector() != null) 
                return input;
            else 
            {
                var fm = new FlattenedMatricesAsVector(input.Matrices);
                float[] fv = fm.FlattenAllMatricesAndCopyUgh();
                AvxColumnVector ffv = new AvxColumnVector(fv);
                return ffv.ToTensor();
            }
        }
    }
}
