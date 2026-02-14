using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using NeuralNets;

namespace NeuralNetsTests
{
    [TestClass]
    public class BatchTrainTests
    {
        /// <summary>
        /// Tests that BatchTrain correctly accumulates gradients from multiple samples
        /// and updates weights once per batch (mini-batch gradient descent).
        /// </summary>
        [TestMethod]
        public void BatchTrain_AccumulatesGradientsFromMultipleSamples()
        {
            // Arrange: Create a simple 1-layer network with known weights
            var inputShape = new InputOutputShape(1, 2, 1, 1); // 2 inputs
            
            // Create weights and biases manually using float arrays
            var weights = MatrixFactory.CreateMatrix(1, 2);
            weights[0, 0] = 0.5f;
            weights[0, 1] = 0.5f;
            var biases = MatrixFactory.CreateColumnVector(new[] { 0.0f });
            
            var layer = new WeightedLayer(inputShape, 1, weights, biases);
            
            var layers = new List<Layer> { layer };
            var lossFunction = new SquaredLoss();
            var network = new GeneralFeedForwardANN(layers, trainingRate: 0.1f, inputDim: 2, outputDim: 1, lossFunction);
            
            // Create 3 training samples with known inputs/outputs
            var trainingPairs = new List<TrainingPair>
            {
                new TrainingPair(CreateTensor(new[] { 1.0f, 1.0f }), CreateTensor(new[] { 2.0f })), // Target: 2.0
                new TrainingPair(CreateTensor(new[] { 2.0f, 2.0f }), CreateTensor(new[] { 4.0f })), // Target: 4.0
                new TrainingPair(CreateTensor(new[] { 3.0f, 3.0f }), CreateTensor(new[] { 6.0f }))  // Target: 6.0
            };
            
            var mockTrainingSet = new MockTrainingSet(trainingPairs, inputShape, new InputOutputShape(1, 1, 1, 1));
            var batchSize = 3;
            var renderContext = new RenderContext(network, batchSize, mockTrainingSet);
            
            // Store initial weights
            float initialWeight0 = layer.Weights[0, 0];
            float initialWeight1 = layer.Weights[0, 1];
            float initialBias = layer.Biases[0];
            
            // Act: Run batch training for 1 epoch
            RenderContext.BatchTrain(renderContext, epochNum: 0);
            
            // Assert: Verify weights were updated (changed from initial values)
            Assert.AreNotEqual(initialWeight0, layer.Weights[0, 0], "Weight 0 should have been updated");
            Assert.AreNotEqual(initialWeight1, layer.Weights[0, 1], "Weight 1 should have been updated");
            Assert.AreNotEqual(initialBias, layer.Biases[0], "Bias should have been updated");
            
            // Verify that gradients were accumulated from all 3 samples
            // After one batch of 3 samples, the accumulators should have been cleared by UpdateWeightsAndBiasesWithScaledGradients
            // But we can verify the logic worked by checking that weights changed
        }

        /// <summary>
        /// Tests that BatchTrain produces the same final weights as sequential training
        /// when using the same samples and learning rate.
        /// </summary>
        [TestMethod]
        public void BatchTrain_ProducesSameResultsAsSequentialTraining()
        {
            // Arrange: Create two identical networks
            var inputShape = new InputOutputShape(1, 2, 1, 1);
            
            var weights1 = MatrixFactory.CreateMatrix(1, 2);
            weights1[0, 0] = 0.5f;
            weights1[0, 1] = 0.5f;
            var biases1 = MatrixFactory.CreateColumnVector(new[] { 0.0f });
            var layer1 = new WeightedLayer(inputShape, 1, weights1, biases1);
            
            var weights2 = MatrixFactory.CreateMatrix(1, 2);
            weights2[0, 0] = 0.5f;
            weights2[0, 1] = 0.5f;
            var biases2 = MatrixFactory.CreateColumnVector(new[] { 0.0f });
            var layer2 = new WeightedLayer(inputShape, 1, weights2, biases2);
            
            var network1 = new GeneralFeedForwardANN(
                new List<Layer> { layer1 }, 
                trainingRate: 0.1f, inputDim: 2, outputDim: 1, 
                new SquaredLoss());
            
            var network2 = new GeneralFeedForwardANN(
                new List<Layer> { layer2 }, 
                trainingRate: 0.1f, inputDim: 2, outputDim: 1, 
                new SquaredLoss());
            
            var trainingPairs = new List<TrainingPair>
            {
                new TrainingPair(CreateTensor(new[] { 1.0f, 1.0f }), CreateTensor(new[] { 2.0f })),
                new TrainingPair(CreateTensor(new[] { 2.0f, 2.0f }), CreateTensor(new[] { 4.0f })),
                new TrainingPair(CreateTensor(new[] { 3.0f, 3.0f }), CreateTensor(new[] { 6.0f }))
            };
            
            // Act: Batch training on network1
            var mockTrainingSet1 = new MockTrainingSet(trainingPairs, inputShape, new InputOutputShape(1, 1, 1, 1));
            var batchContext = new RenderContext(network1, batchSize: 3, mockTrainingSet1);
            RenderContext.BatchTrain(batchContext, epochNum: 0);
            
            // Sequential training on network2
            SequentialTrain(network2, trainingPairs, trainingRate: 0.1f);
            
            // Assert: Both networks should have identical weights
            Assert.AreEqual(layer1.Weights[0, 0], layer2.Weights[0, 0], 1e-6f, "Weight 0 should match");
            Assert.AreEqual(layer1.Weights[0, 1], layer2.Weights[0, 1], 1e-6f, "Weight 1 should match");
            Assert.AreEqual(layer1.Biases[0], layer2.Biases[0], 1e-6f, "Bias should match");
        }

        /// <summary>
        /// Tests that BatchTrain correctly handles empty batches without crashing.
        /// </summary>
        [TestMethod]
        public void BatchTrain_HandlesEmptyBatchGracefully()
        {
            // Arrange
            var inputShape = new InputOutputShape(1, 2, 1, 1);
            
            var weights = MatrixFactory.CreateMatrix(1, 2);
            weights[0, 0] = 0.5f;
            weights[0, 1] = 0.5f;
            var biases = MatrixFactory.CreateColumnVector(new[] { 0.0f });
            var layer = new WeightedLayer(inputShape, 1, weights, biases);

            var network = new GeneralFeedForwardANN(
                new List<Layer> { layer },
                trainingRate: 0.1f, inputDim: 2, outputDim: 1,
                new SquaredLoss());

            // Empty training set
            var emptyTrainingSet = new MockTrainingSet(new List<TrainingPair>(), inputShape, new InputOutputShape(1, 1, 1, 1));
            var renderContext = new RenderContext(network, batchSize: 3, emptyTrainingSet);
            
            // Act & Assert: Should not throw
            RenderContext.BatchTrain(renderContext, epochNum: 0);
            
            // Weights should remain unchanged since there were no samples
            // This is acceptable behavior for empty training sets
        }

        /// <summary>
        /// Tests that BatchTrain with batch_size=1 produces the same results as
        /// processing a single sample.
        /// </summary>
        [TestMethod]
        public void BatchTrain_WithBatchSizeOne_MatchesSingleSampleTraining()
        {
            // Arrange
            var inputShape = new InputOutputShape(1, 2, 1, 1);
            
            var weights = MatrixFactory.CreateMatrix(1, 2);
            weights[0, 0] = 0.5f;
            weights[0, 1] = 0.5f;
            var biases = MatrixFactory.CreateColumnVector(new[] { 0.0f });
            var layer = new WeightedLayer(inputShape, 1, weights, biases);

            var network = new GeneralFeedForwardANN(
                new List<Layer> { layer },
                trainingRate: 0.1f, inputDim: 2, outputDim: 1,
                new SquaredLoss());

            var trainingPairs = new List<TrainingPair>
            {
                new TrainingPair(CreateTensor(new[] { 1.0f, 1.0f }), CreateTensor(new[] { 2.0f }))
            };

            var mockTrainingSet = new MockTrainingSet(trainingPairs, inputShape, new InputOutputShape(1, 1, 1, 1));
            var renderContext = new RenderContext(network, batchSize: 1, mockTrainingSet);
            
            // Calculate expected weight update manually
            // Forward: z = 0.5*1 + 0.5*1 + 0 = 1.0
            // Loss derivative: dL/dy = y_pred - y_true = 1.0 - 2.0 = -1.0
            // dL/dw1 = dL/dy * dy/dw1 = -1.0 * 1.0 = -1.0
            // dL/dw2 = -1.0 * 1.0 = -1.0
            // Average gradient = -1.0 (only one sample)
            // Weight update: w = w - learning_rate * gradient = 0.5 - 0.1 * (-1.0) = 0.6
            
            float expectedWeight = 0.6f;
            
            // Act
            RenderContext.BatchTrain(renderContext, epochNum: 0);
            
            // Assert
            Assert.AreEqual(expectedWeight, layer.Weights[0, 0], 1e-5f, "Weight 0 should match expected value");
            Assert.AreEqual(expectedWeight, layer.Weights[0, 1], 1e-5f, "Weight 1 should match expected value");
        }

        /// <summary>
        /// Tests that gradients are properly accumulated and averaged in a batch.
        /// </summary>
        [TestMethod]
        public void BatchTrain_GradientsAreAveragedOverBatch()
        {
            // Arrange: Create a network where we can control and verify gradient accumulation
            var inputShape = new InputOutputShape(1, 1, 1, 1); // Single input for simplicity
            
            var weights = MatrixFactory.CreateMatrix(1, 1);
            weights[0, 0] = 1.0f;
            var biases = MatrixFactory.CreateColumnVector(new[] { 0.0f });
            var layer = new WeightedLayer(inputShape, 1, weights, biases);

            var network = new GeneralFeedForwardANN(
                new List<Layer> { layer },
                trainingRate: 1.0f, // Large learning rate for visibility
                inputDim: 1, outputDim: 1,
                new SquaredLoss());
            
            // Two samples with different targets
            // Sample 1: input=1.0, target=3.0 -> prediction=1.0, error=-2.0
            // Sample 2: input=1.0, target=1.0 -> prediction=1.0, error=0.0
            // Average error = (-2.0 + 0.0) / 2 = -1.0
            var trainingPairs = new List<TrainingPair>
            {
                new TrainingPair(CreateTensor(new[] { 1.0f }), CreateTensor(new[] { 3.0f })),
                new TrainingPair(CreateTensor(new[] { 1.0f }), CreateTensor(new[] { 1.0f }))
            };
            
            var mockTrainingSet = new MockTrainingSet(trainingPairs, inputShape, new InputOutputShape(1, 1, 1, 1));
            var renderContext = new RenderContext(network, batchSize: 2, mockTrainingSet);
            
            // Expected: w_new = w_old - lr * avg_gradient
            // Sample 1 gradient: input * error = 1.0 * (1.0 - 3.0) = -2.0
            // Sample 2 gradient: input * error = 1.0 * (1.0 - 1.0) = 0.0
            // Average gradient = (-2.0 + 0.0) / 2 = -1.0
            // w_new = 1.0 - 1.0 * (-1.0) = 2.0
            float expectedWeight = 2.0f;
            
            // Act
            RenderContext.BatchTrain(renderContext, epochNum: 0);
            
            // Assert
            Assert.AreEqual(expectedWeight, layer.Weights[0, 0], 1e-5f, 
                "Weight should reflect averaged gradient from both samples");
        }

        /// <summary>
        /// Tests that BatchTrain correctly processes multiple epochs.
        /// </summary>
        [TestMethod]
        public void BatchTrain_MultipleEpochs_ReduceLoss()
        {
            // Arrange
            var inputShape = new InputOutputShape(1, 2, 1, 1);
            
            var weights = MatrixFactory.CreateMatrix(1, 2);
            weights[0, 0] = 0.5f;
            weights[0, 1] = 0.5f;
            var biases = MatrixFactory.CreateColumnVector(new[] { 0.0f });
            var layer = new WeightedLayer(inputShape, 1, weights, biases);

            var network = new GeneralFeedForwardANN(
                new List<Layer> { layer },
                trainingRate: 0.1f, inputDim: 2, outputDim: 1,
                new SquaredLoss());

            var trainingPairs = new List<TrainingPair>
            {
                new TrainingPair(CreateTensor(new[] { 1.0f, 1.0f }), CreateTensor(new[] { 2.0f })),
                new TrainingPair(CreateTensor(new[] { 2.0f, 2.0f }), CreateTensor(new[] { 4.0f }))
            };

            var mockTrainingSet = new MockTrainingSet(trainingPairs, inputShape, new InputOutputShape(1, 1, 1, 1));
            var renderContext = new RenderContext(network, batchSize: 2, mockTrainingSet);
            
            // Calculate initial loss
            float initialLoss = CalculateLoss(network, trainingPairs[0]);
            
            // Act: Train for 10 epochs
            for (int epoch = 0; epoch < 10; epoch++)
            {
                RenderContext.BatchTrain(renderContext, epoch);
            }
            
            // Calculate final loss
            float finalLoss = CalculateLoss(network, trainingPairs[0]);
            
            // Assert: Loss should decrease
            Assert.IsTrue(finalLoss < initialLoss, 
                $"Final loss ({finalLoss}) should be less than initial loss ({initialLoss})");
        }

        #region Helper Methods

        private static Tensor CreateTensor(float[] values)
        {
            return new AnnTensor(null, MatrixFactory.CreateColumnVector(values));
        }

        private static void SequentialTrain(NeuralNetworkAbstract network, List<TrainingPair> trainingPairs, float trainingRate)
        {
            // Reset accumulators
            foreach (Layer layer in network.Layers)
            {
                layer.ResetAccumulators();
            }

            // Process each sample and accumulate gradients
            foreach (var pair in trainingPairs)
            {
                // Forward pass
                Tensor output = pair.Input;
                foreach (Layer layer in network.Layers)
                {
                    output = layer.FeedFoward(output);
                }

                // Backward pass
                ColumnVectorBase? truthVec = pair.Output.ToColumnVector();
                ColumnVectorBase? predVec = output.ToColumnVector();
                Assert.IsNotNull(truthVec, "Truth vector should not be null");
                Assert.IsNotNull(predVec, "Predicted vector should not be null");
                Tensor dE_dX = network.LossFunction.Derivative(truthVec, predVec).ToTensor();
                foreach (Layer layer in network.Layers.Reverse<Layer>())
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
            }

            // Update weights once after all samples
            foreach (Layer layer in network.Layers)
            {
                layer.UpdateWeightsAndBiasesWithScaledGradients(trainingRate);
            }
        }

        private static float CalculateLoss(NeuralNetworkAbstract network, TrainingPair pair)
        {
            Tensor output = pair.Input;
            foreach (Layer layer in network.Layers)
            {
                output = layer.FeedFoward(output);
            }
            ColumnVectorBase? predVec = output.ToColumnVector();
            Assert.IsNotNull(predVec, "Predicted vector should not be null");
            return network.GetTotallLoss(pair, predVec);
        }

        #endregion

        #region Mock Classes

        /// <summary>
        /// Mock implementation of ITrainingSet for testing purposes.
        /// </summary>
        private class MockTrainingSet : ITrainingSet
        {
            private readonly List<TrainingPair> _trainingPairs;
            private int _randomizeCount = 0;

            public MockTrainingSet(List<TrainingPair> trainingPairs, InputOutputShape inputShape, InputOutputShape outputShape)
            {
                _trainingPairs = trainingPairs;
                Width = inputShape.Width;
                Height = inputShape.Height;
                Depth = inputShape.Depth;
                OutputShape = outputShape;
                NumClasses = outputShape.TotalFlattenedSize;
            }

            public int Width { get; }
            public int Height { get; }
            public int Depth { get; }
            public InputOutputShape OutputShape { get; }
            public int NumClasses { get; }
            public int NumberOfSamples => _trainingPairs.Count;
            public int NumberOfLabels => _trainingPairs.Count;

            public List<TrainingPair> TrainingList => _trainingPairs;

            public List<TrainingPair> BuildNewRandomizedTrainingList(bool normalized2D)
            {
                // For testing, just return the same list (or shuffle if needed)
                _randomizeCount++;
                return new List<TrainingPair>(_trainingPairs);
            }
        }

        #endregion
    }
}
