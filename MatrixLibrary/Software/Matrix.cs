using System.Text;
using System.Diagnostics;
using MatrixLibrary.BaseClasses;
using System.Runtime.CompilerServices;

namespace MatrixLibrary
{

    public class Matrix2D : MatrixBase
    {
        public MatrixBackend Backend => MatrixBackend.Software;

        public Matrix2D(int r, int c)
        {
            Rows = r;
            Cols = c;
            Mat = new float[Rows, Cols];
        }

        public Matrix2D(float[,] m)
        {
            Rows = m.GetLength(0);
            Cols = m.GetLength(1);
            this.Mat = m; // no deep copy, better not change my matrix dude!
        }

        public override Matrix2D Log()
        {
            Matrix2D logMat = new Matrix2D(this.Rows, this.Cols);
            for (int r = 0; r < this.Rows; ++r)
            {
                for (int c = 0; c < this.Cols; ++c)
                {
                    logMat[r, c] = (float)Math.Log(this[r, c]);
                }
            }
            return logMat;
        }

        public override void SetDiagonal(float diagonalValue)
        {
            if (Rows != Cols)
                throw new InvalidOperationException("Can't set diagonal of a non-square matrix");

            for (int i = 0; i < Rows; i++)
                Mat[i, i] = diagonalValue;
        }

        public override Matrix2D Add(MatrixBase b)
        {
            if (this.Rows != b.Rows || this.Cols != b.Cols)            
                throw new ArgumentException("bad dimensions in Matrix.add");            

            Matrix2D res = new Matrix2D(Rows, Cols);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    res.Mat[r, c] = this[r, c] + b[r, c];
                }
            }

            return res;
        }

        // I'm on the left of 'm'
        public override Matrix2D Multiply(MatrixBase m)
        {
            if (this.Cols == m.Rows)
            {
                Matrix2D res = new Matrix2D(this.Rows, m.Cols);
                for (int r = 0; r < Rows; r++)
                {
                    // multiply my horizontal vector times m's vertical vector
                    // my r and it's C
                    int leftR = r;
                    for (int rightCol = 0; rightCol < m.Cols; rightCol++)
                    {
                        res.Mat[leftR, rightCol] = DoRowTimesColumn(leftR, rightCol, m);
                    }
                }

                return res;
            }
            else
            {
                throw new ArgumentException("Bad dimensions");
            }
        }

/*        public RowVector RowTimesMatrix(RowVector left)
        {
            if (left.Size == this.Rows)
            {
                float[] vector = new float[left.Size];
                for (int c = 0; c < this.Cols; c++)
                {
                    vector[c] = DoRowVectorTimesColumn(left, c);
                }
                return new RowVector(vector);
            }
            else
            {
                throw new ArgumentOutOfRangeException("Bad dimensions");
            }
        }
*/

        public override ColumnVector MatrixTimesColumn(ColumnVectorBase colVec)
        {
            if (this.Cols == colVec.Size)
            {
                float[] vector = new float[this.Rows];
                for (int r = 0; r < Rows; r++)
                {
                    vector[r] = DoRowTimesColumnVector(r, colVec);
                }

                return new ColumnVector(vector);
            }
            else
            {
                throw new ArgumentException("Bad dimensions");
            }
        }

        public override Matrix2D Multiply(float scalar)
        {
            Matrix2D res = new Matrix2D(Rows, Cols);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    res.Mat[r, c] = scalar * this[r, c];
                }
            }
            return res;
        }

        public override Matrix2D Subtract(MatrixBase b)
        {
            if (!HasSameDimensions(b))
                throw new ArgumentException("bad dimensions in Matrix.subtract");

            // this minus b
            if (this.Cols == b.Cols && this.Rows == b.Rows)
            {
                Matrix2D res = new Matrix2D(this.Rows, b.Cols);
                for (int r = 0; r < Rows; r++)
                {
                    for (int c = 0; c < this.Cols; c++)
                    {
                        res[r, c] = this[r, c] - b[r, c];
                    }
                }
                return res;
            }
            else
            {
                throw new ArgumentOutOfRangeException("Bad dimensions");
            }
        }

        public override Matrix2D HadamardProduct(MatrixBase b)
        {
            if (!this.HasSameDimensions(b))
                throw new ArgumentException("bad dimensions in Matrix.HadamardProduct");

            Matrix2D res = new Matrix2D(Rows, Cols);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < this.Cols; c++)
                {
                    res.Mat[r, c] = this.Mat[r, c] * b.Mat[r, c];
                }
            }
            return res;
        }

        private bool HasSameDimensions(MatrixBase b) => (Rows == b.Rows) && (Cols == b.Cols);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float DoRowTimesColumn(int myRow, int rightMatrixCol, MatrixBase rightMatrix)
        {
            float result = 0;
            for (int i = 0; i < Cols; i++)
                result += this[myRow, i] * rightMatrix[i, rightMatrixCol];
            return result;
        }

        private float DoRowTimesColumnVector(int myRow, ColumnVectorBase colVec)
        {
            Debug.Assert(colVec.Size == this.Cols);
            float product = 0;
            for(int c = 0; c < Cols; c++)
            {
                product += this.Mat[myRow, c] * colVec[c];
            }
            return product;
        }

        private float DoRowVectorTimesColumn(RowVectorBase rowVec, int myCol)
        {
            Debug.Assert(rowVec.Size == this.Rows);
            float cum = 0;
            for (int i = 0; i < rowVec.Size; i++)
            {
                cum += rowVec[i] * this.Mat[i, myCol];
            }
            return cum;
        }

        public override Matrix2D GetTransposedMatrix()
        {
            Matrix2D mt = new Matrix2D(this.Cols, this.Rows);
            for (int c = 0; c < this.Cols; c++)
            {
                for (int r = 0; r < this.Rows; r++)
                {
                    mt.Mat[c, r] = this.Mat[r, c];
                }
            }
            return mt;
        }

        public void Print()
        {
            StringBuilder str = new StringBuilder();
            for (int i = 0; i < this.Rows; ++i)
            {
                for (int j = 0; j < this.Cols; ++j)
                {
                    str.Append(this.Mat[i, j].ToString("F3").PadLeft(8) + " ");
                }
                str.AppendLine();
            }
            Console.Write(str);
        }

        public override MatrixBase Add(float scalar)
        {
            Matrix2D res = new Matrix2D(Rows, Cols);
            for (int r = 0; r < Rows; r++)
                for (int c = 0; c < Cols; c++)
                    res.Mat[r, c] = this[r, c] + scalar;

            return res;
        }

        public override Matrix2D Convolution(MatrixBase kernel)
        {
            // Valid convolution: kernel stays entirely within the input matrix
            // Output size: (rows - kernel_rows + 1) x (cols - kernel_cols + 1)
            
            if (kernel.Rows > this.Rows || kernel.Cols > this.Cols)
                throw new ArgumentException("Kernel dimensions cannot exceed matrix dimensions");
            
            int outputRows = this.Rows - kernel.Rows + 1;
            int outputCols = this.Cols - kernel.Cols + 1;
            
            Matrix2D result = new Matrix2D(outputRows, outputCols);
            
            // Slide kernel across the matrix
            for (int outRow = 0; outRow < outputRows; outRow++)
            {
                for (int outCol = 0; outCol < outputCols; outCol++)
                {
                    float sum = 0;
                    
                    // Apply kernel at current position
                    for (int kRow = 0; kRow < kernel.Rows; kRow++)
                    {
                        for (int kCol = 0; kCol < kernel.Cols; kCol++)
                        {
                            int inRow = outRow + kRow;
                            int inCol = outCol + kCol;
                            sum += this[inRow, inCol] * kernel[kRow, kCol];
                        }
                    }
                    
                    result[outRow, outCol] = sum;
                }
            }
            
            return result;
        }

        public override Matrix2D ConvolutionFull(MatrixBase kernel)
        {
            // Full convolution: kernel can extend past the edges (padding with implied zeros)
            // Output size: (rows + kernel_rows - 1) x (cols + kernel_cols - 1)
            
            int outputRows = this.Rows + kernel.Rows - 1;
            int outputCols = this.Cols + kernel.Cols - 1;
            
            Matrix2D result = new Matrix2D(outputRows, outputCols);
            
            // For full convolution, we center the kernel such that it can extend beyond edges
            // The kernel's top-left corner can start at: -(kernel.Rows - 1) to (this.Rows - 1)
            // But we iterate through output positions
            
            int kernelCenterRow = kernel.Rows / 2;
            int kernelCenterCol = kernel.Cols / 2;
            
            for (int outRow = 0; outRow < outputRows; outRow++)
            {
                for (int outCol = 0; outCol < outputCols; outCol++)
                {
                    float sum = 0;
                    
                    // For each output position, calculate which input elements contribute
                    // The kernel is applied centered at various positions
                    for (int kRow = 0; kRow < kernel.Rows; kRow++)
                    {
                        for (int kCol = 0; kCol < kernel.Cols; kCol++)
                        {
                            // Calculate input position
                            // When kernel center is at outRow, outCol in output space
                            // Kernel element [kRow, kCol] aligns with input at:
                            int inRow = outRow - (kernel.Rows - 1) + kRow;
                            int inCol = outCol - (kernel.Cols - 1) + kCol;
                            
                            // Check bounds - if outside matrix, it's zero (padding)
                            if (inRow >= 0 && inRow < this.Rows && inCol >= 0 && inCol < this.Cols)
                            {
                                sum += this[inRow, inCol] * kernel[kRow, kCol];
                            }
                            // Else: implied zero padding, nothing to add
                        }
                    }
                    
                    result[outRow, outCol] = sum;
                }
            }
            
            return result;
        }

        public override MatrixBase Transpose()
        {
            return GetTransposedMatrix();
        }
    }
}
