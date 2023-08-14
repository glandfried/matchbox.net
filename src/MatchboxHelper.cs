using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Learners;
using Microsoft.ML.Probabilistic.Math;

namespace MatchboxHelper
{
    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;

    public class MatchboxMapping<T> : Microsoft.ML.Probabilistic.Learners.Mappings.IStarRatingRecommenderMapping<T, Tuple<string, string, int>, string, string, int, NoFeatureSource, Vector>
    {
        public virtual IEnumerable<Tuple<string, string, int>> GetInstances(T instanceSource)
        { throw new NotImplementedException(); }

        public string GetUser(T instanceSource, Tuple<string, string, int> instance)
        { return instance.Item1; }

        public string GetItem(T instanceSource, Tuple<string, string, int> instance)
        { return instance.Item2; }

        public int GetRating(T instanceSource, Tuple<string, string, int> instance)
        { return instance.Item3; }

        public Microsoft.ML.Probabilistic.Learners.IStarRatingInfo<int> GetRatingInfo(T instanceSource)
        { return new Microsoft.ML.Probabilistic.Learners.StarRatingInfo(0, 5); }

        public Vector GetUserFeatures(Microsoft.ML.Probabilistic.Learners.NoFeatureSource featureSource, string user)
        { throw new NotImplementedException(); }

        public Vector GetItemFeatures(Microsoft.ML.Probabilistic.Learners.NoFeatureSource featureSource, string item)
        { throw new NotImplementedException(); }
    }

    [Serializable]
    public class CsvMapping : MatchboxMapping<string>
    {
        private char sep;
        public CsvMapping() { this.sep = ','; }
        public CsvMapping(char separator) { this.sep = separator; }
        override public IEnumerable<Tuple<string, string, int>> GetInstances(string instanceSource)
        {
            using (StreamReader sr = File.OpenText(instanceSource))
            {
                string s = String.Empty;
                int i = 0;
                while ((s = sr.ReadLine()) != null)
                {
                    string[] split = s.Split(new[] { this.sep });
                    Console.WriteLine("Read line {0}", i);
                    i++;
                    yield return Tuple.Create(split[0], split[1], Convert.ToInt32(split[2]));
                }
            }
            /*
            foreach (string line in File.ReadLines(instanceSource))
            {
                string[] split = line.Split(new[] { this.sep });
                yield return Tuple.Create(split[0], split[1], Convert.ToInt32(split[2]));
            }
            */
        }
    }

    // https://stackoverflow.com/questions/10297124/how-to-combine-more-than-two-generic-lists-in-c-sharp-zip
    public static class MyFunkyExtensions
    {
        public static IEnumerable<TResult> ZipThree<T1, T2, T3, TResult>(
            IEnumerable<T1> source,
            IEnumerable<T2> second,
            IEnumerable<T3> third,
            Func<T1, T2, T3, TResult> func)
        {
            using (var e1 = source.GetEnumerator())
            using (var e2 = second.GetEnumerator())
            using (var e3 = third.GetEnumerator())
            {
                while (e1.MoveNext() && e2.MoveNext() && e3.MoveNext())
                    yield return func(e1.Current, e2.Current, e3.Current);
            }
        }
    }

    [Serializable]
    public class DataframeMapping : MatchboxMapping<Tuple<Dictionary<string, Array>, Array>>
    {
        public DataframeMapping() {Console.WriteLine("Creating Dataframe Mapping");}

        override public IEnumerable<Tuple<string, string, int>> GetInstances(Tuple<Dictionary<string, Array>, Array> tup)
        {
            /*
            for (int i=0; i<df["1"].Length;i++) 
            {
                yield return Tuple.Create(df["1"][i], df["2"][i], df["3"][i]);
            }
            return df["1"].Zip(df["2"], (first, second) => Tuple.Create(first,second))
                          .Zip(df["3"], (first, second) => Tuple.Create(first.Item1, first.Item2, second));
            */
            using (var e1 = (IEnumerator<string>) tup.Item1["userId"].GetEnumerator())
            using (var e2 = (IEnumerator<string>) tup.Item1["movieId"].GetEnumerator())
            using (var e3 = (IEnumerator<string>) tup.Item2.GetEnumerator())
            {
                while (e1.MoveNext() && e2.MoveNext() && e3.MoveNext())
                    yield return Tuple.Create(e1.Current, e2.Current, Convert.ToInt32(e3.Current));
            }
        }
    }

    public static class MatchboxCsvWrapper 
    {
        public static Microsoft.ML.Probabilistic.Learners.IMatchboxRecommender<string, string, string, RatingDistribution, Microsoft.ML.Probabilistic.Learners.NoFeatureSource> 
        Create(CsvMapping mapping) {
            return Microsoft.ML.Probabilistic.Learners.MatchboxRecommender.Create(mapping);
        }

        public static Microsoft.ML.Probabilistic.Learners.IMatchboxRecommender<Tuple<Dictionary<string, Array>, Array>, string, string, RatingDistribution, Microsoft.ML.Probabilistic.Learners.NoFeatureSource> 
        Create(DataframeMapping mapping) {
            Console.WriteLine("Creating Matchbox recommender");
            return Microsoft.ML.Probabilistic.Learners.MatchboxRecommender.Create(mapping);
        }

        public static Tuple<Dictionary<string, Array>, Array> MakeTuple(Dictionary<string, Array> a, Array b)
        {
            Console.WriteLine("string tuple conversion");
            return Tuple.Create(a, b);
        }
    }
}
