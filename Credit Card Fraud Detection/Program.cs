using CreditCardFraudDetection.Data;
using Microsoft.ML;
using System;

namespace CreditCardFraudDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            //var modelBuilder = new ModelBuilder().Create(@"../../../Data/creditcard.csv").Save(@"../../../Data/CreditCardFraudDetection.zip");

            MLContext mlContext = new MLContext(seed: 1);

            string modelZip = @"../../../Data/CreditCardFraudDetection.zip";
            string datacreditcardCsv = @"../../../Data/creditcard.csv";

            // Loading model
            ITransformer mlModel = mlContext.Model.Load(modelZip, out DataViewSchema inputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Get test data-points
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                           path: datacreditcardCsv,
                                           hasHeader: true,
                                           separatorChar: ',',
                                           allowQuoting: true,
                                           allowSparse: false);

            var sampleData = mlContext.Data.TrainTestSplit(dataView, 0.1);

            var testSet = mlContext.Data
                .CreateEnumerable<ModelInput>(sampleData.TestSet, false);

            foreach (ModelInput m in testSet)
            {
                //Feed test data
                ModelOutput predictionResult = predEngine.Predict(m);

                Console.WriteLine($"Actual value: {m.Class} | Predicted value: {predictionResult.Prediction}");
            }

            Console.ReadKey();
        }
    }
}
