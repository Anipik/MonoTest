using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace monotest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("hello");
           
             string _dataPath = @"C:\git\machinelearning\test\data\adult.tiny.with-schema.txt";
                var ml = new MLContext(seed: 1);
            // Pipeline

            var input = ml.Data.ReadFromTextFile(_dataPath, new[] {
                            new TextLoader.Column("Label", DataKind.BL, 0),
                            new TextLoader.Column("CatFeatures", DataKind.TX,
                                new [] {
                                    new TextLoader.Range() { Min = 1, Max = 8 },
                                }),
                            new TextLoader.Column("NumFeatures", DataKind.R4,
                                new [] {
                                    new TextLoader.Range() { Min = 9, Max = 14 },
                                }),
            }, hasHeader: true);

            var estimatorPipeline = ml.Transforms.Categorical.OneHotEncoding("CatFeatures")
                .Append(ml.Transforms.Normalize("NumFeatures"))
                .Append(ml.Transforms.Concatenate("Features", "NumFeatures", "CatFeatures"))
                .Append(ml.Clustering.Trainers.KMeans("Features"))
                .Append(ml.Transforms.Concatenate("Features", "Features", "Score"))
                .Append(ml.BinaryClassification.Trainers.LogisticRegression());

            var model = estimatorPipeline.Fit(input);
            Console.ReadLine();
        }
    }
}
