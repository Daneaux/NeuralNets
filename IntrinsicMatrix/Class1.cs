using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics;
using System.Text;
using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace IntrinsicMatrix
{
#if false
    public class Matrix
    {

        private float[,] Mat;
        public Matrix() { }
        public Matrix(float[,] mat) { Mat = mat; }

        public Matrix(int rows, int cols)
        {
            Rows = rows;
            Cols = cols;
        }

        public int Rows { get; private set; }
        public int Cols { get; private set; }

        private Matrix Add_avx(Matrix b)
        {
            if (this.Rows != b.Rows || this.Cols != b.Cols)
            {
                throw new ArgumentException("bad dimensions in Matrix.add");
            }

            Matrix res = new Matrix(Rows, Cols);

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
#endif
}
