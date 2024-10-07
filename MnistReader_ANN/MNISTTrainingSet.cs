using MatrixLibrary;
using NeuralNets;
using System.Diagnostics;

namespace MnistReader_ANN
{
    public class MNISTTrainingSet : ITrainingSet
    {
        public MNISTTrainingSet()
        {
            this.NumClasses = 10;

            int width = 0;
            int height = 0;
            int numLabels = 0;
            int numSamples = 0;

            MnistReader.GetMNISTTrainingMetaData(out numSamples, out numLabels, out width, out height);

            Width = width;
            Height = height;
            Depth = 1;
            NumberOfLabels = numLabels;
            NumberOfSamples = numSamples;
            OutputShape = new InputOutputShape(Width, Height, Depth, 1);
        }

        public int Width { get; }
        public int Height { get; }
        public int Depth { get; }
        public int NumClasses { get; private set; }
        public int NumberOfSamples { get; private set; }
        public int NumberOfLabels { get; private set; }
        public InputOutputShape OutputShape { get; }

        public List<TrainingPair> TrainingList { get; private set; }
        private List<Normalized2DImage> ImageList { get; set; }

        public List<TrainingPair> BuildNewRandomizedTrainingList()
        {
            Random rnd = new Random();
            if (this.ImageList == null)
            {
                this.ImageList = MnistReader.ReadNormTrainingData().ToList();
                Debug.Assert(ImageList.Count == 60000);
            }
            List<TrainingPair> trainingPairs = new List<TrainingPair>((int)ImageList.Count);
            foreach (Normalized2DImage image in ImageList)
            {
                trainingPairs.Add(TrainingPairFromImage(image));
            }            

            this.TrainingList = trainingPairs.OrderBy(x => rnd.Next()).ToList();
            return this.TrainingList;
        }

        private TrainingPair TrainingPairFromImage(Normalized2DImage image)
        {
            AvxMatrix image2d = new AvxMatrix(image.Data);
            AvxColumnVector outputVector = OneHotEncodeLabelData(image);
            TrainingPair trainingPair = new TrainingPair(image2d.ToTensor(), outputVector.ToTensor());
            return trainingPair;
        }

        private AvxColumnVector OneHotEncodeLabelData(Normalized2DImage image)
        {
            // convert the label data (0,1,2, ...) into a columnvector. if the label is 7 (ie: byte == 7), then set the 7th float to 1.0
            float[] labelData = new float[this.NumClasses];
            labelData[(int)image.Label] = 1;
            return new AvxColumnVector(labelData);
        }

        private static AvxColumnVector ImageDataToColumnVector(Normalized2DImage image)
        {
            return null; // new AvxColumnVector(image.Data);
            /*
            // convert the image data into a columnvector
            float[] imageData = new float[image.Size];
            int i = 0;
            foreach (byte b in image.Data)
            {
                imageData[i++] = (float)b;
            }

            return new AvxColumnVector(imageData);*/
        }
    }
}
