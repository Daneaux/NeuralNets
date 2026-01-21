using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using Microsoft.Diagnostics.Tracing.Parsers.MicrosoftWindowsTCPIP;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    public class NormalizationLayer : Layer
    {
        public NormalizationLayer(InputOutputShape inputShape, int nodeCount, int randomSeed = 12341324) : base(inputShape, nodeCount, randomSeed)
        {
            OutputShape = inputShape;
        }

        public override InputOutputShape OutputShape { get; }

        internal struct normMatMetaData
        {
            public float Mean;
            public float Stddev;
            public MatrixBase NormalizedMatrix;
        }
        private List<normMatMetaData> normMatMetaDataList;

        internal class normVectorMetaData
        {
            public double Mean { get; set; }
            public double Stddev { get; set; }
            public ColumnVectorBase NormalizedVector { get; set; }
        }
        private List<normVectorMetaData> normVectorMetaDataList = new List<normVectorMetaData>();


        public override Tensor FeedFoward(Tensor input)
        {
            // mean over all input. std deviation over all input
            // for each input (x - mean) / (std + epsilon)
            ColumnVectorBase vec = input.ToColumnVector();
            if(vec != null)
                return doVectorFeedForward(vec);
            else
                return doMatrixFeedForward(input.Matrices);
        }

        private Tensor doVectorFeedForward(ColumnVectorBase input)
        {
            this.normMatMetaDataList = new List<normMatMetaData>();
            float epsilon = 0.000000001f;

            WelfordAlgorithm soMean = new WelfordAlgorithm(input);

            var mean = (float)soMean.Mean;
            var stdDev = (float)soMean.StandardDeviation;

            ColumnVectorBase normVector = (input + -mean) * (float)(1.0f / stdDev);
            normVectorMetaDataList.Add(new normVectorMetaData { Mean = mean, Stddev = stdDev, NormalizedVector = normVector });

            return normVector.ToTensor();
        }
        private Tensor doMatrixFeedForward(List<MatrixBase> input)
        {
            this.normMatMetaDataList = new List<normMatMetaData>();
            float epsilon = 0.000000001f;
            List<MatrixBase> normalizedMatrices = new List<MatrixBase>();
            foreach (MatrixBase mat in input)
            {
                float mean = this.incrementalAverage(mat);
                float stdDev = this.incrementalStdDeviation(mat, mean) + epsilon;
                MatrixBase normMat = (mat + -mean) * (float)(1.0f / stdDev);
                normalizedMatrices.Add(normMat);
                normMatMetaDataList.Add(new normMatMetaData { Mean = mean, Stddev = stdDev, NormalizedMatrix = normMat });
            }
            return normalizedMatrices.ToTensor();
        }

        // reference:
        // https://neuralthreads.medium.com/layer-normalization-and-how-to-compute-its-jacobian-for-backpropagation-55a549d5936f
        public override Tensor BackPropagation(Tensor dE_dY)
        {
            float epsilon = 0.000000001f;
            ColumnVectorBase inputVec =  dE_dY.ToColumnVector();
            
            // I have no idea what it means if we have multiple normMatMetaData's and one input vector ... TODO
            Debug.Assert(inputVec != null);
            Debug.Assert(this.normMatMetaDataList.Count == 1);

            List<MatrixBase> jacobians = new List<MatrixBase>();
            List<ColumnVectorBase> results = new List<ColumnVectorBase>();
            foreach (normVectorMetaData meta in this.normVectorMetaDataList)
            {
                int N = meta.NormalizedVector.Size;
                MatrixBase IN = MatrixFactory.CreateMatrix(N, N);
                IN.SetDiagonal((float)N);
                double oneOverN = 1.0 / (double)N;
                double oneOverStdDev = 1.0 / (meta.Stddev + epsilon);
                double oneOverNTimes_OneOverStdDev = oneOverN * oneOverStdDev;
                MatrixBase p1 = (IN + -1.0f) * (float)(oneOverNTimes_OneOverStdDev);  // N * I - 1 / Nq

                double OneOverNTimes_OneOverStdDevPow3 = oneOverN * oneOverStdDev * oneOverStdDev * oneOverStdDev;

                var p2 = meta.NormalizedVector + (float) -meta.Mean;
                // P2 is huge, before we make it even bigger, let's pre-reduce it
                p2 = p2 * (float)OneOverNTimes_OneOverStdDevPow3;
                MatrixBase p3 = p2.OuterProduct(p2);  // == p2 dot p2_T
/*                MatrixBase p3_1 = p3 * (float)oneOverN;
                MatrixBase p3_2 = p3_1 * (float)oneOverStdDev;
                 p3_2 = p3_1 * (float)oneOverStdDev;
                 p3_2 = p3_1 * (float)oneOverStdDev;
*/                //MatrixBase p4 = p3 * (float)(OneOverNTimes_OneOverStdDevPow3);  // (x-mu) dot (x-mu)T / Nq^3
                MatrixBase p4 = p3;
                MatrixBase Jacobian = p1 - p4;
                jacobians.Add(Jacobian);
                ColumnVectorBase dydx = Jacobian * inputVec;
                results.Add(dydx);
            }

            return results[0].ToTensor();
        }

        public Tensor BackPropagation_(Tensor dE_dY)
        {
            float epsilon = 0.000000001f;
            List<MatrixBase> result = new List<MatrixBase>();
            foreach (normMatMetaData meta in this.normMatMetaDataList)
            {
                int N = meta.NormalizedMatrix.Rows;
                MatrixBase IN = MatrixFactory.CreateMatrix(N, N);
                IN.SetDiagonal(N);
                MatrixBase p1 = (IN + -1.0f) * (N * meta.Stddev + epsilon);
                MatrixBase p2 = meta.NormalizedMatrix + -meta.Mean;
                MatrixBase p3 = p2.GetTransposedMatrix();

            }
            return result.ToTensor();
        }

        private float incrementalAverage(ColumnVectorBase vec)
        {
            int sz = vec.Size;
            double avg = 0;
            for (int r = 0; r < sz; r++)
                avg += (vec[r] - avg) / sz;

            return (float)avg;
        }

        private float incrementalStdDeviation(ColumnVectorBase vec, float knownAverage)
        {
            int sz = vec.Size;
            double squareddiff = 0;
            double sumSqDiff = 0;

            for (int r = 0; r < sz; r++)
            {
                squareddiff = (vec[r] - knownAverage) * (vec[r] - knownAverage);
                sumSqDiff += squareddiff;
            }

            double variance = sumSqDiff / sz;
            return (float)Math.Sqrt(variance);
        }


        private float incrementalAverage(MatrixBase mat)
        {
            int sz = mat.TotalSize;
            double avg = 0;
            for (int r = 0; r < mat.Rows; r++)
                for (int c = 0; c < mat.Cols; c++)
                    avg += (mat[r, c] - avg) / sz;

            return (float)avg;
        }

        private float incrementalStdDeviation(MatrixBase mat, float knownAverage)
        {
            int sz = mat.TotalSize;
            double squareddiff = 0;
            double sumSqDiff = 0;

            for (int r = 0; r < mat.Rows; r++)
                for (int c = 0; c < mat.Cols; c++)
                {
                    squareddiff = (mat[r, c] - knownAverage) * (mat[r, c] - knownAverage);
                    sumSqDiff += squareddiff;
                }

            double variance = sumSqDiff / sz;
            return (float)Math.Sqrt(variance);
        }

    }

}
