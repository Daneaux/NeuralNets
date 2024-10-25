using MatrixLibrary;
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
            public AvxMatrix NormalizedMatrix;
        }
        private List<normMatMetaData> normMatMetaDataList;

        internal class normVectorMetaData
        {
            public double Mean { get; set; }
            public double Stddev { get; set; }
            public AvxColumnVector NormalizedVector { get; set; }
        }
        private List<normVectorMetaData> normVectorMetaDataList = new List<normVectorMetaData>();


        public override Tensor FeedFoward(Tensor input)
        {
            // mean over all input. std deviation over all input
            // for each input (x - mean) / (std + epsilon)
            AvxColumnVector vec = input.ToAvxColumnVector();
            if(vec != null)
                return doVectorFeedForward(vec);
            else
                return doMatrixFeedForward(input.Matrices);
        }

        private Tensor doVectorFeedForward(AvxColumnVector input)
        {
            this.normMatMetaDataList = new List<normMatMetaData>();
            float epsilon = 0.000000001f;

            WelfordAlgorithm soMean = new WelfordAlgorithm(input);

            var mean = (float)soMean.Mean;
            var stdDev = (float)soMean.StandardDeviation;

            AvxColumnVector normVector = (input + -mean) * (float)(1.0f / stdDev);
            normVectorMetaDataList.Add(new normVectorMetaData { Mean = mean, Stddev = stdDev, NormalizedVector = normVector });

            return normVector.ToTensor();
        }
        private Tensor doMatrixFeedForward(List<AvxMatrix> input)
        {
            this.normMatMetaDataList = new List<normMatMetaData>();
            float epsilon = 0.000000001f;
            List<AvxMatrix> normalizedMatrices = new List<AvxMatrix>();
            foreach (AvxMatrix mat in input)
            {
                float mean = this.incrementalAverage(mat);
                float stdDev = this.incrementalStdDeviation(mat, mean) + epsilon;
                AvxMatrix normMat = (mat + -mean) * (float)(1.0f / stdDev);
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
            AvxColumnVector inputVec = dE_dY.ToAvxColumnVector();
            
            // I have no idea what it means if we have multiple normMatMetaData's and one input vector ... TODO
            Debug.Assert(inputVec != null);
            Debug.Assert(this.normMatMetaDataList.Count == 1);

            List<AvxMatrix> jacobians = new List<AvxMatrix>();
            List<AvxColumnVector> results = new List<AvxColumnVector>();
            foreach (normVectorMetaData meta in this.normVectorMetaDataList)
            {
                int N = meta.NormalizedVector.Size;
                AvxMatrix IN = new AvxMatrix(N, N);
                IN.SetDiagonal((float)N);
                double oneOverN = 1.0 / (double)N;
                double oneOverStdDev = 1.0 / (meta.Stddev + epsilon);
                double oneOverNTimes_OneOverStdDev = oneOverN * oneOverStdDev;
                AvxMatrix p1 = (IN + -1.0f) * (float)(oneOverNTimes_OneOverStdDev);  // N * I - 1 / Nq

                double OneOverNTimes_OneOverStdDevPow3 = oneOverN * oneOverStdDev * oneOverStdDev * oneOverStdDev;

                var p2 = meta.NormalizedVector + (float) -meta.Mean;
                // P2 is huge, before we make it even bigger, let's pre-reduce it
                p2 = p2 * (float)OneOverNTimes_OneOverStdDevPow3;
                AvxMatrix p3 = p2.OuterProduct(p2);  // == p2 dot p2_T
/*                AvxMatrix p3_1 = p3 * (float)oneOverN;
                AvxMatrix p3_2 = p3_1 * (float)oneOverStdDev;
                 p3_2 = p3_1 * (float)oneOverStdDev;
                 p3_2 = p3_1 * (float)oneOverStdDev;
*/                //AvxMatrix p4 = p3 * (float)(OneOverNTimes_OneOverStdDevPow3);  // (x-mu) dot (x-mu)T / Nq^3
                AvxMatrix p4 = p3;
                AvxMatrix Jacobian = p1 - p4;
                jacobians.Add(Jacobian);
                AvxColumnVector dydx = Jacobian * inputVec;
                results.Add(dydx);
            }

            return results[0].ToTensor();
        }

        public Tensor BackPropagation_(Tensor dE_dY)
        {
            float epsilon = 0.000000001f;
            List<AvxMatrix> result = new List<AvxMatrix>();
            foreach (normMatMetaData meta in this.normMatMetaDataList)
            {
                int N = meta.NormalizedMatrix.Rows;
                AvxMatrix IN = new AvxMatrix(N, N);
                IN.SetDiagonal(N);
                AvxMatrix p1 = (IN + -1.0f) * (N * meta.Stddev + epsilon);
                AvxMatrix p2 = meta.NormalizedMatrix + -meta.Mean;
                AvxMatrix p3 = p2.GetTransposedMatrix();

            }
            return result.ToTensor();
        }

        private float incrementalAverage(AvxColumnVector vec)
        {
            int sz = vec.Size;
            double avg = 0;
            for (int r = 0; r < sz; r++)
                avg += (vec[r] - avg) / sz;

            return (float)avg;
        }

        private float incrementalStdDeviation(AvxColumnVector vec, float knownAverage)
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


        private float incrementalAverage(AvxMatrix mat)
        {
            int sz = mat.TotalSize;
            double avg = 0;
            for (int r = 0; r < mat.Rows; r++)
                for (int c = 0; c < mat.Cols; c++)
                    avg += (mat[r, c] - avg) / sz;

            return (float)avg;
        }

        private float incrementalStdDeviation(AvxMatrix mat, float knownAverage)
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
