using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ILikeDieHard
{
    public class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var trainingData = new List<MoviePreferenceInput>();
            var dieHardLover = new MoviePreferenceInput
            {
                StarWarsScore = 8,
                ArmageddonScore = 10,
                SleeplessInSeattleScore = 1,
                ILikeDieHard = true
            };
            var dieHardHater = new MoviePreferenceInput
            {
                StarWarsScore = 1,
                ArmageddonScore = 1,
                SleeplessInSeattleScore = 9,
                ILikeDieHard = false
            };

            for (var i = 0; i < 100; i++)
            {
                trainingData.Add(dieHardLover);
                trainingData.Add(dieHardHater);
            }

            IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);
            ITransformer model;
            if (!File.Exists("./diehard-model.zip") && !File.Exists("./diehard-pipeline.zip"))
            {
                model = TrainNewModel(mlContext, trainingDataView);
            }
            else
            {
                model = RetrainModel(mlContext, trainingDataView);
            }

            var input1 = new MoviePreferenceInput
            {
                StarWarsScore = 7,
                ArmageddonScore = 9,
                SleeplessInSeattleScore = 0
            };
            var input2 = new MoviePreferenceInput
            {
                StarWarsScore = 0,
                ArmageddonScore = 0,
                SleeplessInSeattleScore = 10
            };

            PredictionEngine<MoviePreferenceInput, LikeDieHardPrediction> predictionEngine =
                mlContext.Model.CreatePredictionEngine<MoviePreferenceInput, LikeDieHardPrediction>(model);
            var prediction = predictionEngine.Predict(input1);
            Console.WriteLine($"First user loves Die Hard: {prediction.Prediction}");
            prediction = predictionEngine.Predict(input2);
            Console.WriteLine($"Second user loves Die Hard: {prediction.Prediction}");
        }

        private static ITransformer RetrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            DataViewSchema dataPrepPipelineSchema, modelSchema;
            var trainedModel = mlContext.Model.Load("./diehard-model.zip", out modelSchema);
            var dataPrePipeline = mlContext.Model.Load("./diehard-pipeline.zip", out dataPrepPipelineSchema);

            IDataView transformedData = dataPrePipeline.Transform(trainingDataView);
            IEnumerable<ITransformer> chain = trainedModel as IEnumerable<ITransformer>;
            ISingleFeaturePredictionTransformer<object> predictionTransformer = chain.Last() as ISingleFeaturePredictionTransformer<object>;
            var originalModelParameters = predictionTransformer.Model as LinearBinaryModelParameters;

            var model = dataPrePipeline
                .Append(mlContext
                    .BinaryClassification
                    .Trainers
                    .AveragedPerceptron(labelColumnName: "ILikeDieHard", numberOfIterations: 10, featureColumnName: "Features")
                    .Fit(transformedData, originalModelParameters));

            mlContext.Model.Save(model, trainingDataView.Schema, "./diehard-model.zip");

            return model;
        }

        private static ITransformer TrainNewModel(MLContext mlContext, IDataView trainingDataView)
        {
            var dataPrepPipeline = mlContext
                .Transforms
                .Concatenate(
                    outputColumnName:"Features",
                    "StarWarsScore",
                    "ArmageddonScore",
                    "SleeplessInSeattleScore")
                .AppendCacheCheckpoint(mlContext);

            var prepPipeline = dataPrepPipeline.Fit(trainingDataView);

            mlContext.Model.Save(prepPipeline, trainingDataView.Schema, "./diehard-pipeline.zip");

            var trainer = dataPrepPipeline.Append(mlContext
                .BinaryClassification
                .Trainers
                .AveragedPerceptron(labelColumnName: "ILikeDieHard", numberOfIterations: 10, featureColumnName: "Features"));

            var preprocessedData = prepPipeline.Transform(trainingDataView);
            var model = trainer.Fit(preprocessedData);
            mlContext.Model.Save(model, trainingDataView.Schema, "./diehard-model.zip");
            return model;
        }
    }

    class MoviePreferenceInput
    {
        public float StarWarsScore { get; set; }
        public float ArmageddonScore { get; set; }
        public float SleeplessInSeattleScore { get; set; }

        public bool ILikeDieHard { get; set; }
    }

    class LikeDieHardPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }
}
