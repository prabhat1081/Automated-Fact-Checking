from imblearn.over_sampling import SMOTE
from imblearn import under_sampling
from imblearn import over_sampling
from imblearn import ensemble



d = {
	# Under samplers
	"cluster_centroids" : 	under_sampling.ClusterCentroids(),	#Perform under-sampling by generating centroids based on clustering methods.
	"condesedNN":			under_sampling.CondensedNearestNeighbour(),	#Class to perform under-sampling based on the condensed nearest neighbour method.
	"ENN":					under_sampling.EditedNearestNeighbours(),	#Class to perform under-sampling based on the edited nearest neighbour method.
	"repeatededitedNN":		under_sampling.RepeatedEditedNearestNeighbours(),	#Class to perform under-sampling based on the repeated edited nearest neighbour method.
	"instance_hardness":	under_sampling.InstanceHardnessThreshold(),	#Class to perform under-sampling based on the instance hardness threshold.
	"nearmiss":				under_sampling.NearMiss(),	#Class to perform under-sampling based on NearMiss methods.
	"nbrcleaning":			under_sampling.NeighbourhoodCleaningRule(),	#Class performing under-sampling based on the neighbourhood cleaning rule.
	"1sided":				under_sampling.OneSidedSelection(),	#Class to perform under-sampling based on one-sided selection method.
	"random":				under_sampling.RandomUnderSampler(),	#Class to perform random under-sampling.
	"tomek":				under_sampling.TomekLinks(),

	# Over samplers

	"adasyn":				over_sampling.ADASYN(),#	Perform over-sampling using ADASYN.
	"random_over":			over_sampling.RandomOverSampler(),#	Class to perform random over-sampling.
	"smote":				over_sampling.SMOTE(random_state=42), #	Class to perform over-sampling using SMOTE.

	# Ensemble

	"balance_cascade":		ensemble.BalanceCascade(classifier="linear-svm", n_max_subset=10),	#Create an ensemble of balanced sets by iteratively under-sampling the imbalanced dataset using an estimator.
	"easy_ensemble":		ensemble.EasyEnsemble(random_state=557, n_subsets=50)	#Create an ensemble sets by iteratively applying random under-sampling.


}



def get_sampler(name):
	return d[name]