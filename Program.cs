using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

using Range = Microsoft.ML.Probabilistic.Models.Range;

// Define counts  
int numUsers = 50;  
int numItems = 10;  
int numTraits = 2;  
Variable<int> numObservations = Variable.Observed(100);  
int numLevels = 2;  

// Define ranges  
Range user = new Range(numUsers);  
Range item = new Range(numItems);  
Range trait = new Range(numTraits);  
Range observation = new Range(numObservations);  
Range level = new Range(numLevels);

// It’s important to note that in Infer.NET this type of an array is defined as:
// Variable.Array(Variable.Array<double>(numColumns), numRows)  //(notar el numcolumns priemro)
// The return type of this operation is
// VariableArray<VariableArray<double>, double[][]>  
// Define latent variables  
var userTraits = Variable.Array(Variable.Array<double>(trait), user);  
var itemTraits = Variable.Array(Variable.Array<double>(trait), item);  
var userBias = Variable.Array<double>(user);  
var itemBias = Variable.Array<double>(item);  
var userThresholds = Variable.Array(Variable.Array<double>(level), user);

// Define priors  
var userTraitsPrior = Variable.Array(Variable.Array<Gaussian>(trait), user);  
var itemTraitsPrior = Variable.Array(Variable.Array<Gaussian>(trait), item);  
var userBiasPrior = Variable.Array<Gaussian>(user);  
var itemBiasPrior = Variable.Array<Gaussian>(item);  
var userThresholdsPrior = Variable.Array(Variable.Array<Gaussian>(level), user);

// Define latent variables statistically  
userTraits[user][trait] = Variable<double>.Random(userTraitsPrior[user][trait]);  
itemTraits[item][trait] = Variable<double>.Random(itemTraitsPrior[item][trait]);  
userBias[user] = Variable<double>.Random(userBiasPrior[user]);  
itemBias[item] = Variable<double>.Random(itemBiasPrior[item]);  
userThresholds[user][level] = Variable<double>.Random(userThresholdsPrior[user][level]);

// Initialise priors  
Gaussian traitPrior = Gaussian.FromMeanAndVariance(0.0, 1.0);  
Gaussian biasPrior = Gaussian.FromMeanAndVariance(0.0, 1.0);  
userTraitsPrior.ObservedValue = Util.ArrayInit(numUsers, u => Util.ArrayInit(numTraits, t => traitPrior));  
itemTraitsPrior.ObservedValue = Util.ArrayInit(numItems, i => Util.ArrayInit(numTraits, t => traitPrior));  
userBiasPrior.ObservedValue = Util.ArrayInit(numUsers, u => biasPrior);  
itemBiasPrior.ObservedValue = Util.ArrayInit(numItems, i => biasPrior);  
userThresholdsPrior.ObservedValue = Util.ArrayInit(numUsers, u =>  
        Util.ArrayInit(numLevels, l => Gaussian.FromMeanAndVariance(l - numLevels / 2.0 + 0.5, 1.0)));

// Declare training data variables  
var userData = Variable.Array<int>(observation);  
var itemData = Variable.Array<int>(observation);  
var ratingData = Variable.Array(Variable.Array<bool>(level), observation);

// Model  
using (Variable.ForEach(observation)) {  
    VariableArray<double> productos = Variable.Array<double>(trait);  
    productos[trait] = userTraits[userData[observation]][trait] * itemTraits[itemData[observation]][trait];  

    //We set the noise level to 0.1 because of the small amount of data we work with, but in practise probably a choice of 1.0 might yield better results.
    double affinityNoiseVariance = 0.1; 
    double thresholdsNoiseVariance = 0.1;

    Variable<double> _bias = (userBias[userData[observation]] + itemBias[itemData[observation]]);  
    Variable<double> _affinity = (_bias + Variable.Sum(productos)); 
    Variable<double> _noisyAffinity = Variable.GaussianFromMeanAndVariance(_affinity, affinityNoiseVariance);  

    VariableArray<double> _noisyThresholds = Variable.Array<double>(level);  
    _noisyThresholds[level] = Variable.GaussianFromMeanAndVariance(userThresholds[userData[observation]][level], thresholdsNoiseVariance);  
    ratingData[observation][level] = _noisyAffinity > _noisyThresholds[level];
}

// Allow EP to process the product factor as if running VMP  
// as in Stern, Herbrich, Graepel paper. 
InferenceEngine engine = new InferenceEngine();  
engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));

// Run inference  
var userTraitsPosterior = engine.Infer<Gaussian[][]>(userTraits);  
var itemTraitsPosterior = engine.Infer<Gaussian[][]>(itemTraits);  
var userBiasPosterior = engine.Infer<Gaussian[]>(userBias);  
var itemBiasPosterior = engine.Infer<Gaussian[]>(itemBias);  
var userThresholdsPosterior = engine.Infer<Gaussian[][]>(userThresholds);  

// Feed in the inferred posteriors as the new priors  
userTraitsPrior.ObservedValue = userTraitsPosterior;  
itemTraitsPrior.ObservedValue = itemTraitsPosterior;  
userBiasPrior.ObservedValue = userBiasPosterior;  
itemBiasPrior.ObservedValue = itemBiasPosterior;  
userThresholdsPrior.ObservedValue = userThresholdsPosterior;

// Make a prediction  
numObservations.ObservedValue = 1;  
userData.ObservedValue = new int[] { 5 };  
itemData.ObservedValue = new int[] { 6 };  
ratingData.ClearObservedValue();  

Bernoulli[] predictedRating = engine.Infer<Bernoulli[][]>(ratingData)[0];  
Console.WriteLine("Predicted rating:");  
foreach (var rating in predictedRating) Console.WriteLine(rating);

// Break symmetry and remove ambiguity in the traits  
// TODO: no entendi esto
for (int i = 0; i < numTraits; i++) {    // Assume that numTraits < numItems  
    for (int j = 0; j < numTraits; j++) {  
        itemTraitsPrior.ObservedValue[i][j] = Gaussian.PointMass(0);  
    }  
    itemTraitsPrior.ObservedValue[i][i] = Gaussian.PointMass(1);  
}

// Sample model parameters from the priors  
Rand.Restart(12347);  
double[][] userTraits = Util.ArrayInit(numUsers, u => Util.ArrayInit(numTraits, t => userTraitsPrior[u][t].Sample()));  
double[][] itemTraits = Util.ArrayInit(numItems, i => Util.ArrayInit(numTraits, t => itemTraitsPrior[i][t].Sample()));  
double[] userBias = Util.ArrayInit(numUsers, u => userBiasPrior[u].Sample());  
double[] itemBias = Util.ArrayInit(numItems, i => itemBiasPrior[i].Sample());  
double[][] userThresholds = Util.ArrayInit(numUsers, u => Util.ArrayInit(numLevels, l => userThresholdsPrior[u][l].Sample()));

// Repeat the model with fixed parameters  
HashSet<int> visited = new HashSet<int>();  
for (int observation = 0; observation < numObservations; observation++) {
      int user = Rand.Int(numUsers);  
      int item = Rand.Int(numItems); int userItemPairID = user * numItems + item; // pair encoding  
    if (visited.Contains(userItemPairID)) // duplicate generated {  
        observation--; // reject pair continue;  
    }  
    visited.Add(userItemPairID);
    double[] products = Util.ArrayInit(numTraits, t => userTraits[user][t] * itemTraits[item][t]);
    double bias = userBias[user] + itemBias[item];
    double affinity = bias + products.Sum();
    double noisyAffinity = new Gaussian(affinity, affinityNoiseVariance).Sample();
    double[] noisyThresholds = Util.ArrayInit(numLevels, l => new Gaussian(userThresholds[user][l], thresholdsNoiseVariance).Sample());  

    generatedUserData[observation] = user;  
    generatedItemData[observation] = item;  
    generatedRatingData[observation] = Util.ArrayInit(numLevels, l => noisyAffinity > noisyThresholds[l]);  
}
