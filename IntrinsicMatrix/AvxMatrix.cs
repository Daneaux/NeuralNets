using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace IntrinsicMatrix
{
    public class AvxColumnVector
    {
        private readonly float[] column;
        public int Size { get { return column.Length; } }

        public float this[int a] => column[a];
        public float[] Column { get { return column; } }

        public AvxColumnVector(float[] column)
        {
            this.column = column;
        }
        public AvxColumnVector(int size)
        {
            this.column = new float[size];
        }
    }

    public class AvxMatrix
    {
        private readonly float[,] Mat;
        public float this[int r, int c]
        {
            get { return this.Mat[r, c]; }
        }

        public AvxMatrix(float[,] mat) 
        {
            Mat = mat; 
            Rows = Mat.GetLength(0);
            Cols = Mat.GetLength(1);
        }

        public AvxMatrix(int rows, int cols)
        {
            Rows = rows;
            Cols = cols;
            Mat = new float[rows, cols];
        }

        public int Rows { get; private set; }
        public int Cols { get; private set; }

        public static AvxMatrix operator+(AvxMatrix lhs, AvxMatrix rhs) => lhs.AddAvx512(rhs);

        // [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe AvxMatrix AddAvx512(AvxMatrix b)
        {
            const int floatsPerVector = 16;
            int size = Rows * Cols;
            int numVectors = size / floatsPerVector;
            int remainingElements = size % floatsPerVector;

            AvxMatrix result = new AvxMatrix(Rows, Cols);

            fixed (float* m1 = this.Mat,
                          m2 = b.Mat,
                          d1 = result.Mat)
            {
                float* mat1 = m1;
                float* mat2 = m2;
                float* dest = d1;

                for (int i = 0; i < numVectors; i++, mat1 += 16, mat2 += 16, dest += 16)
                {
                    Vector512<float> v1 = Vector512.Load<float>(mat1);
                    Vector512<float> v2 = Vector512.Load<float>(mat2);
                    Vector512<float> sum = Avx512F.Add(v1, v2);
                    Vector512.Store<float>(sum, dest);
                }

                // do remainder
                for (int i = 0; i < remainingElements; i++, dest++, mat1++, mat2++)
                {
                    *dest = *mat1 + *mat2;
                }
            }

            return result;
        }

        public unsafe AvxColumnVector MatrixTimesColumn(AvxColumnVector b)
        {
            const int floatsPerVector = 16;
            int numVectorsPerRow = Cols / floatsPerVector;
            int remainingColumns = Cols % floatsPerVector;

            AvxColumnVector result = new AvxColumnVector(Rows);

            fixed (float* m1 = this.Mat,
                          col = b.Column,
                          resCol = result.Column)
            {
                float* mat1 = m1;
                float* col1 = col;
                float* destCol = resCol;

                Vector512<float> colVector = Vector512.Load<float>(col1);
                for (int r = 0; r < Rows; r++, destCol++)
                {
                    col1 = col;             // restart at the top of the column vector
                    float v1DotCol = 0;
                    for (int c = 0; c < numVectorsPerRow; c++, mat1 += 16, col1 += 16)
                    {
                        Vector512<float> v1 = Vector512.Load<float>(mat1);
                        v1DotCol += Vector512.Dot(v1, colVector);
                    }

                    // do remainder
                    for (int i = 0; i < remainingColumns; i++, mat1++, col1++)
                    {
                        v1DotCol += (*mat1) * (*col1);
                    }
                    *destCol = v1DotCol;
                }
            }

            return result;
        }


        private AvxMatrix Add_avx(AvxMatrix b)
        {
            if (this.Rows != b.Rows || this.Cols != b.Cols)
            {
                throw new ArgumentException("bad dimensions in Matrix.add");
            }

            AvxMatrix res = new AvxMatrix(Rows, Cols);

            int vectorSize = Vector512<float>.Count; // AVX-512 processes 8 doubles at a time (512 bits / 64 bits per float)

            for (int r = 0; r < Rows; r++)
            {
                int c = 0;

                // Check if the CPU supports AVX-512
                if (Avx512F.IsSupported)
                {
                    // Process in chunks of vectorSize
                    for (; c <= Cols - vectorSize; c += vectorSize)
                    {
                        /*                        Vector512<float> vec1 = Vector512.Create(
                            this.Mat[r, c], this.Mat[r, c + 1], this.Mat[r, c + 2], this.Mat[r, c + 3],
                            this.Mat[r, c + 4], this.Mat[r, c + 5], this.Mat[r, c + 6], this.Mat[r, c + 7]
                            );
                                                Vector512<float> vec2 = Vector512.Create(
                            b.Mat[r, c], b.Mat[r, c + 1], b.Mat[r, c + 2], b.Mat[r, c + 3],
                            b.Mat[r, c + 4], b.Mat[r, c + 5], b.Mat[r, c + 6], b.Mat[r, c + 7]
                            );*/
                        //var vec1 = Vector512.Load(&Mat[r, c]);
                        //var vec2 = Vector512.Load(&b.Mat[r, c]);
                        //var vecRes = vec1 + vec2;
                        //vecRes.Store(&res.Mat[r, c]);
                        // Store the result back into the result matrix (safe method)

                        //vecRes.CopyTo(res.Mat, r * Cols + c);
                    }
                }

                // Process any remaining elements
                /*                for (; c < Cols; c++)
                                {
                                    res.Mat[r, c] = this[r, c] + b[r, c];
                                }*/
            }

            return res;
        }
    }
}
