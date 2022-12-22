using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Learners;
using Microsoft.ML.Probabilistic.Math;

namespace CsvMapping
{
    [Serializable]
    public class CsvMapping : Microsoft.ML.Probabilistic.Learners.Mappings.IStarRatingRecommenderMapping<string, Tuple<string, string, int>, string, string, int, NoFeatureSource, Vector>
    {
        public IEnumerable<Tuple<string, string, int>> GetInstances(string instanceSource)
        {
            foreach (string line in File.ReadLines(instanceSource))
            {
                string[] split = line.Split(new[] { ',' });
                yield return Tuple.Create(split[0], split[1], Convert.ToInt32(split[2]));
            }
        }

        public string GetUser(string instanceSource, Tuple<string, string, int> instance)
        { return instance.Item1; }

        public string GetItem(string instanceSource, Tuple<string, string, int> instance)
        { return instance.Item2; }

        public int GetRating(string instanceSource, Tuple<string, string, int> instance)
        { return instance.Item3; }

        public Microsoft.ML.Probabilistic.Learners.IStarRatingInfo<int> GetRatingInfo(string instanceSource)
        { return new Microsoft.ML.Probabilistic.Learners.StarRatingInfo(0, 5); }

        public Vector GetUserFeatures(Microsoft.ML.Probabilistic.Learners.NoFeatureSource featureSource, string user)
        { throw new NotImplementedException(); }

        public Vector GetItemFeatures(Microsoft.ML.Probabilistic.Learners.NoFeatureSource featureSource, string item)
        { throw new NotImplementedException(); }
    }
}
