# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
import sys
import os
import clr
from System import Array, Double, Int32, Type, Converter, Tuple
import System
from System import Console
from System import Array
from System import IO
#from utils import plot_graph

netpackagedir = System.IO.Path.GetDirectoryName(System.IO.FileInfo(__file__).FullName)+"/MatchboxHelper"
#sys.path.append(r'/opt/dotNet/libs')
sys.path.append(netpackagedir)

#add path to infer.net dlls .................................
clr.AddReference("Microsoft.ML.Probabilistic")
clr.AddReference("Microsoft.ML.Probabilistic.Compiler")
clr.AddReference("Microsoft.ML.Probabilistic.Learners")
clr.AddReference("Microsoft.ML.Probabilistic.Learners.Recommender")
clr.AddReference("MatchboxHelper")
#clr.AddReference(os.path.normpath("./CsvMapping"))

#---import all namespaces----------------
import Microsoft.ML.Probabilistic
import Microsoft.ML.Probabilistic.Models
import Microsoft.ML.Probabilistic.Distributions
import Microsoft.ML.Probabilistic.Collections
import Microsoft.ML.Probabilistic.Factors
import Microsoft.ML.Probabilistic.Math
from Microsoft.ML.Probabilistic import * 

#---------import all classes and methods from above namespaces-----------
from Microsoft.ML.Probabilistic.Distributions import *
from Microsoft.ML.Probabilistic.Models import *
from Microsoft.ML.Probabilistic.Collections import *
from Microsoft.ML.Probabilistic.Factors import *
from Microsoft.ML.Probabilistic.Math import *

import Microsoft.ML.Probabilistic.Learners
from Microsoft.ML.Probabilistic.Learners import *

import MatchboxHelper
from MatchboxHelper import *
import MatchboxPropio
from MatchboxPropio import *