using System.Diagnostics;

namespace MatrixLibrary
{
    public class InputOutputShape
    {
        public InputOutputShape(int x, int y, int z, int count)
        {
            Debug.Assert(x >= 1 && y >= 1 && z >= 1 && count >= 1);
            Width = x;
            Height = y;
            Depth = z;
            Count = count;
        }
        public int Width { get; }
        public int Height { get; }
        public int Depth { get; }
        public int Count { get; }

        public int TotalFlattenedSize => Width * Height * Depth * Count;
    }
}