using NeuralNets;
using NumReaderNetwork;

namespace NeuralNetsTests.ANNTests
{

    // NOTE: not sure any of these work ... 

    [TestClass]
    public class NumberReaderANNTests
    {
        [TestMethod]
        public void TestInitializeNetwork_CreatesCorrectLayers()
        {
            // Arrange
            int inputDim = 786; // Example input dimension
            int hiddenDim = 16;
            int outputDim = 10;
            NumberReaderANN network = new NumberReaderANN(inputDim, hiddenDim, outputDim);

            // Assert
            Assert.AreEqual(2, network.WeightedLayers.Count); // Check if 2 layers are created

            Assert.AreEqual(inputDim, network.WeightedLayers[0].NumNodes);
            Assert.IsInstanceOfType(network.WeightedLayers[0].ActivationFunction, typeof(ReLUActivaction));

            // Verify properties of the output layer
            Assert.AreEqual(network.OutputLayerDim, network.WeightedLayers[1].NumNodes);
            Assert.IsInstanceOfType(network.WeightedLayers[1].ActivationFunction, typeof(SigmoidActivation));
        }

        [TestMethod]
        public void TestFeedForward_CalculatesCorrectOutput()
        {
            // Arrange
            int inputDim = 2;
            int hiddenDim = 2;
            int outputDim = 2;
            NumberReaderANN network = new NumberReaderANN(inputDim, hiddenDim, outputDim);
            network.WeightedLayers = new List<WeightedLayer>()
            {
                new WeightedLayer(2, new ReLUActivaction(), inputDim)
                {
                    Weights = new Matrix(new double[,] { { 0.5, 0.2 }, { 0.1, 0.8 } }),
                    Biases = new ColumnVector([0.3, 0.6])
                },
                new WeightedLayer(1, new SigmoidActivation(), 2)
                {
                    Weights = new Matrix(new double[,] { { 0.7, 0.9 } }),
                    Biases = new ColumnVector(new double[] { 0.1 })
                }
            };
            ColumnVector input = new ColumnVector(new double[] { 0.4, 0.9 });

            // Act
            ColumnVector output = network.FeedForward(input);

            // Assert
            // Calculate the expected output based on the weights and biases
            ColumnVector expectedOutput = new ColumnVector(new double[] { 0.8357 });
            Assert.AreEqual(expectedOutput[0], output[0], 0.001); // Comparing with a tolerance
        }

        [TestMethod]
        public void TestBackProp_UpdatesWeightsAndBiases()
        {
            // Arrange
            int inputDim = 2;
            int hiddenDim = 2;
            int outputDim = 2;
            NumberReaderANN network = new NumberReaderANN(inputDim, hiddenDim, outputDim);
            network.WeightedLayers = new List<WeightedLayer>()
            {
                new WeightedLayer(2, new ReLUActivaction(), inputDim)
                {
                    Weights = new Matrix(new double[,] { { 0.5, 0.2 }, { 0.1, 0.8 } }),
                    Biases = new ColumnVector([0.3, 0.6])
                },
                new WeightedLayer(1, new SigmoidActivation(), 2)
                {
                    Weights = new Matrix(new double[,] { { 0.7, 0.9 } }),
                    Biases = new ColumnVector(new double[] { 0.1 })
                }
            };
            TrainingPair pair = new TrainingPair(new ColumnVector(new double[] { 0.4, 0.9 }), new ColumnVector(new double[] { 0 }));

            // Act
            var predictedOut = network.FeedForward(pair.Input);
            network.BackProp(pair, predictedOut);

            // Assert
            // Check if weights and biases in both layers have been updated
            Assert.AreNotEqual(0.5, network.WeightedLayers[0].Weights[0, 0]); // Example - check a specific weight
            Assert.AreNotEqual(0.1, network.WeightedLayers[1].Biases[0]); // Example - check a specific bias

            // You can add more specific assertions based on the expected changes in weights and biases
        }
    }
}
