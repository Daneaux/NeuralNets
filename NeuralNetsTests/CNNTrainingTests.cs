using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using MnistReader_ANN;
using NeuralNets;

namespace NeuralNetsTests
{
    /// <summary>
    /// Tests for CNN training with proper single-threaded execution
    /// (Parallel training has issues with layer state management)
    /// </summary>
    [TestClass]
    public class CNNTrainingTests
    {
        /// <summary>
        /// Tests CNN training with single-threaded execution.
        /// The CNN architecture works correctly, but parallel training 
        /// has issues with layer state (GradientRouters in PoolingLayer, LastInput in Conv).
        /// </summary>
        [TestMethod]
        public void CNN_SingleThreadedTraining_ReducesLoss()
        {
            // Arrange: Create manual 2D training data
            var trainingPairs = CreateManual2DMNISTData(20);
            
            var inputShape = new InputOutputShape(28, 28, 1, 1);
            
            // Build CNN layers
            var conv1 = new ConvolutionLayer(inputShape, kernelCount: 5, kernelSquareDimension: 4, stride: 1);
            var relu1 = new ReLUActivaction();
            var pool1 = new PoolingLayer(conv1.OutputShape, stride: 2, kernelCount: 5, kernelSquareDimension: 2, kernelDepth: 1);
            var flatten = new FlattenLayer(pool1.OutputShape, nodeCount: 1);
            var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 10);
            var sigmoid = new SigmoidActivation();
            
            var layers = new List<Layer> { conv1, relu1, pool1, flatten, dense, sigmoid };
            
            var network = new GeneralFeedForwardANN(
                layers,
                trainingRate: 0.01f,
                inputDim: 28 * 28,
                outputDim: 10,
                new SquaredLoss());
            
            // Calculate initial loss on first sample
            float initialLoss = CalculateLoss(network, trainingPairs[0]);
            Console.WriteLine($"Initial loss: {initialLoss}");
            
            // Act: Single-threaded training for 5 epochs
            for (int epoch = 0; epoch < 5; epoch++)
            {
                float epochLoss = 0;
                
                foreach (var pair in trainingPairs)
                {
                    // Reset accumulators for each sample
                    foreach (var layer in layers)
                    {
                        layer.ResetAccumulators();
                    }
                    
                    // Forward pass
                    Tensor output = pair.Input;
                    foreach (var layer in layers)
                    {
                        output = layer.FeedFoward(output);
                    }
                    var predicted = output.ToColumnVector();
                    Assert.IsNotNull(predicted);
                    
                    // Calculate loss for this sample
                    float sampleLoss = network.GetTotallLoss(pair, predicted);
                    epochLoss += sampleLoss;
                    
                    // Backward pass
                    var lossDerivative = network.LossFunction.Derivative(
                        pair.Output.ToColumnVector()!, predicted);
                    Tensor dE_dX = lossDerivative.ToTensor();
                    
                    foreach (var layer in layers.Reverse<Layer>())
                    {
                        // All layers (including activation) handle their own derivative computation
                        dE_dX = layer.BackPropagation(dE_dX);
                    }
                    
                    // Update weights immediately (single sample = batch size 1)
                    foreach (var layer in layers)
                    {
                        layer.UpdateWeightsAndBiasesWithScaledGradients(network.LearningRate);
                    }
                }
                
                Console.WriteLine($"Epoch {epoch} average loss: {epochLoss / trainingPairs.Count}");
            }
            
            // Calculate final loss
            float finalLoss = CalculateLoss(network, trainingPairs[0]);
            Console.WriteLine($"Final loss: {finalLoss}");
            
            // Assert: Loss should decrease
            Assert.IsTrue(finalLoss < initialLoss,
                $"Final loss ({finalLoss}) should be less than initial loss ({initialLoss})");
        }

        /// <summary>
        /// Tests that CNN forward and backward passes work correctly with proper sequencing.
        /// This verifies the CNN architecture is sound.
        /// </summary>
        [TestMethod]
        public void CNN_ForwardAndBackward_SequenceWorks()
        {
            // Arrange
            var inputShape = new InputOutputShape(28, 28, 1, 1);
            var conv1 = new ConvolutionLayer(inputShape, kernelCount: 3, kernelSquareDimension: 4, stride: 1);
            var relu1 = new ReLUActivaction();
            var pool1 = new PoolingLayer(conv1.OutputShape, stride: 2, kernelCount: 3, kernelSquareDimension: 2, kernelDepth: 1);
            var flatten = new FlattenLayer(pool1.OutputShape, nodeCount: 1);
            var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 10);
            var sigmoid = new SigmoidActivation();
            
            var layers = new List<Layer> { conv1, relu1, pool1, flatten, dense, sigmoid };
            var network = new GeneralFeedForwardANN(
                layers,
                trainingRate: 0.1f,
                inputDim: 28 * 28,
                outputDim: 10,
                new SquaredLoss());
            
            var trainingPairs = CreateManual2DMNISTData(5);
            var pair = trainingPairs[0];
            
            // Act & Assert: Forward then backward on same sample should work
            try
            {
                // Reset accumulators
                foreach (var layer in layers)
                {
                    layer.ResetAccumulators();
                }
                
                // Forward
                Tensor output = pair.Input;
                foreach (var layer in layers)
                {
                    output = layer.FeedFoward(output);
                }
                var predicted = output.ToColumnVector();
                Assert.IsNotNull(predicted);
                
                // Backward
                var lossDerivative = network.LossFunction.Derivative(
                    pair.Output.ToColumnVector()!, predicted);
                Tensor dE_dX = lossDerivative.ToTensor();
                
                foreach (var layer in layers.Reverse<Layer>())
                {
                    if (layer is IActivationFunction activationLayer)
                    {
                        Tensor activationDerivative = layer.BackPropagation(activationLayer.LastActivation);
                        dE_dX = activationDerivative * dE_dX;
                    }
                    else
                    {
                        dE_dX = layer.BackPropagation(dE_dX);
                    }
                }
                
                // Update weights
                foreach (var layer in layers)
                {
                    layer.UpdateWeightsAndBiasesWithScaledGradients(network.LearningRate);
                }
                
                Assert.IsTrue(true, "Forward-backward sequence completed successfully");
            }
            catch (Exception ex)
            {
                Assert.Fail($"Forward-backward sequence failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Documents that parallel training is available via conditional compilation.
        /// By default, BatchTrain() uses single-threaded execution for safety.
        /// To enable parallel training, define PARALLEL_BATCH_TRAIN in the project settings.
        /// </summary>
        [TestMethod]
        public void CNN_ParallelTraining_IsConditionallyAvailable()
        {
            // BatchTrain() now uses single-threaded execution by default
            // This ensures thread safety with all layer types including CNN layers
            // 
            // To enable parallel training for feed-forward networks (which don't have
            // the state issues that CNN layers have), add this to the .csproj:
            //
            // <PropertyGroup>
            //   <DefineConstants>$(DefineConstants);PARALLEL_BATCH_TRAIN</DefineConstants>
            // </PropertyGroup>
            //
            // Note: Parallel training may fail with CNN layers because:
            // - PoolingLayer stores GradientRouters during FeedForward()
            // - ConvolutionLayer stores LastInput during FeedForward()
            // These are needed during BackPropagation() on the same sample
            
            Assert.Inconclusive(
                "BatchTrain() uses single-threaded execution by default for thread safety. " +
                "To enable parallel execution, define the PARALLEL_BATCH_TRAIN compilation symbol.");
        }

        #region Helper Methods

        private static List<TrainingPair> CreateManual2DMNISTData(int count)
        {
            var pairs = new List<TrainingPair>();
            var random = new Random(42);
            
            for (int i = 0; i < count; i++)
            {
                // Create 28x28 image with random pixel values
                var imageMatrix = MatrixFactory.CreateMatrix(28, 28);
                for (int r = 0; r < 28; r++)
                    for (int c = 0; c < 28; c++)
                    {
                        imageMatrix[r, c] = (float)random.NextDouble();
                    }
                
                var inputTensor = new List<MatrixBase> { imageMatrix }.ToTensor();
                
                // Create one-hot encoded label (random class 0-9)
                int label = random.Next(10);
                float[] labelData = new float[10];
                labelData[label] = 1.0f;
                var labelTensor = new AnnTensor(null, MatrixFactory.CreateColumnVector(labelData));
                
                pairs.Add(new TrainingPair(inputTensor, labelTensor));
            }
            
            return pairs;
        }

        private static float CalculateLoss(NeuralNetworkAbstract network, TrainingPair pair)
        {
            Tensor output = pair.Input;
            foreach (Layer layer in network.Layers)
            {
                output = layer.FeedFoward(output);
            }
            var predVec = output.ToColumnVector();
            Assert.IsNotNull(predVec);
            return network.GetTotallLoss(pair, predVec);
        }

        #endregion
    }
}
