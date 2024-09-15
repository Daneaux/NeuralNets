namespace MatrixLibrary
{
    public interface IRowVector
    {
        public int Size { get; }
        public float[] Row { get; }
        public float this[int i] { get; }
        public float Sum();
    }
}
