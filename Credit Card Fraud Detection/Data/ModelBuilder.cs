using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CreditCardFraudDetection.Data
{
    interface IModelBuilder
    {
        IModelBuilder Create(string dataFilePath);
        IModelBuilder Save(string relativeSavingPath);
    }

    class ModelBuilder : IModelBuilder
    {
        private static MLContext mlContext = new MLContext(seed: 1);
        private static IDataView trainingDataView;
        private static ITransformer mlModel;

        public IModelBuilder Create(string dataFilePath)
        {
            // Load Data
            trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: dataFilePath,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
                "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13",
                "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26",
                "V27", "V28", "Amount");

            // Set the training algorithm 
            IEstimator<ITransformer> trainingPipeline = dataProcessPipeline.Append(mlContext.BinaryClassification.Trainers.LightGbm(labelColumnName: "Class", featureColumnName: "Features"));

            // Evaluate quality of Model
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.BinaryClassification.CrossValidateNonCalibrated(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "Class");
            PrintBinaryClassificationFoldsAverageMetrics(crossValidationResults);

            // Train Model
            mlModel = trainingPipeline.Fit(trainingDataView);
            return this;
        }

        public IModelBuilder Save(string relativeSavingPath = "TrainedModel.zip")
        {
            string absPath = GetAbsPath(relativeSavingPath);
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, trainingDataView.Schema, absPath);
            Console.WriteLine("The model is saved to {0}", absPath);
            return this;
        }

        private static string GetAbsPath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            return Path.Combine(assemblyFolderPath, relativePath);
        }

        private static void PrintBinaryClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var AccuracyValues = metricsInMultipleFolds.Select(m => m.Accuracy);
            var AccuracyAverage = AccuracyValues.Average();
            var AccuraciesStdDeviation = CalculateStandardDeviation(AccuracyValues);
            var AccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(AccuracyValues);


            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Binary Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average Accuracy:    {AccuracyAverage:0.###}  - Standard deviation: ({AccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({AccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");
        }

        private static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            return Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
        }

        private static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            return 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
        }
    }
}
