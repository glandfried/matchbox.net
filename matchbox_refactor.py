#from audioop import bias
#from difflib import Match
import random
from unicodedata import numeric
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import bisect
import itertools
from scipy.optimize import minimize
from scipy.special import rel_entr
from functools import reduce
import math

EPSILON = 1e-3
X_LEFT_BOUND = -15
X_RIGHT_BOUND = 15
X_STEP = 0.1
IMG_DIR = "./img"

class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)

# List of objects that have a "time" attribute. The order of the elements is given by this attribute.
class Timeline(list):
    def __init__(self):
        super(Timeline, self).__init__()
        self.key = lambda x: x.time

    def __repr__(self):
        return super(Timeline, self).__repr__()

    def __eq__(self, other):
        return super(Timeline, self).__eq__(other)
    def __ne__(self, other):
        return not(self == other)

    # TODO: check we're using python lists not numpy arrays because it will be a lot slower
    # TODO: use linked lists here or something that is nicer for insertion
    # Places an element in the Timeline mantaining order by time
    def place(self, e):
        bslindex = bisect.bisect_left(KeyWrapper(self, self.key), e.time)
        self.insert(bslindex, e)

    def ifAbsentPlace(self, e):
        try:
            self.getEventByTime(e.time)
        except ValueError:
            self.place(e)

    # Returns the index of the event in time t
    # Throws ValueError if event that occurred in time t is not present
    #'Locate the leftmost value exactly equal to x'
    def getEventByTime(self, t):
        i = bisect.bisect_left(KeyWrapper(self, self.key), t)
        if i < len(self) and self[i].time == t:
            return self[i]
        raise ValueError

    def append(self, _):
        raise NotImplementedError

    def eventBefore(self, t):
        if self == []:
            raise ValueError
        i = bisect.bisect_left(KeyWrapper(self, self.key), t)
        if i == 0:
            raise ValueError
        return self[i-1]

    def eventAfter(self, t):
        if self == []:
            raise ValueError
        i = bisect.bisect_right(KeyWrapper(self, self.key), t)
        if i >= len(self)-1:
            raise ValueError
        return self[i+1]

class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.sigma2 = sigma**2
        # Non-centered second moment
        self.ncsm = self.mu**2+self.sigma2

    def eval(self, x):
        return norm.pdf(x, self.mu, self.sigma)

    def __mul__(self, other):
        if (other == 0):
            return 0
        if (other == 1):
            return self
        elif (isinstance(other, (int, float))):
            return Gaussian(self.mu*other, self.sigma*other)
        elif (isinstance(other, Uniform)):
            return self
        elif (isinstance(other, Gaussian)):
            return Gaussian.mul(self, other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __matmul__(self, other):
        return self.__mul__(other)
    def __rmatmul__(self, other):
        return self.__rmul__(other)
    def __add__(self, other):
        if (other == 0):
            return self
        # this property is incorrect:
        #elif (isinstance(other, Gaussian)):
        #	return Gaussian(self.mu+other.mu, self.sigma2+other.sigma2)
    def __radd__(self, other):
        if (other == 0):
            return self
        # this property is incorrect:
        #elif (isinstance(other, Gaussian)):
        #	return Gaussian(self.mu+other.mu, self.sigma2+other.sigma2)
    def __repr__(self):
        return f"N({self.mu:.2f}, {self.sigma:.2f})"

    @staticmethod
    def mul(norm1, norm2):
        muStar = norm1.mu/norm1.sigma2 + norm2.mu/norm2.sigma2
        sigma2Star = (1/norm1.sigma2 + 1/norm2.sigma2)**(-1) #c*N(mu, sigma) = N(c*mu, c*sigma) #TODO: agregar al pdf
        #c = Gaussian(norm2.mu, np.sqrt(norm1.sigma2+norm2.sigma2)).eval(norm1.mu) #result should be multiplied by c, but is proportional to not multiplying by it
        return Gaussian(muStar, np.sqrt(sigma2Star))

class Uniform(Gaussian):
    def __init__(self):
        super(Uniform, self).__init__(0, np.inf)

class LatentVariable:
    def __init__(self, name, subject, t0prior):
        self.name = name
        self.subject = subject
        self.t0prior = t0prior
    
    def __hash__(self):
        return hash((self.name, self.subject))
    def __eq__(self, other):
        return self.name == other.name and self.subject == other.subject
    def __ne__(self, other):
        return not(self == other)
    def __repr__(self):
        return f"{self.__class__.__name__} (of {self.subject.__class__.__name__} {self.subject.id})"

class Affinity(LatentVariable):
    def __init__(self, subject, t0prior=None): #TODO: probar inicializar en 0 el mu o algunos negativo y otros positivo
        t0prior = t0prior if t0prior is not None else [Gaussian(0.1 if random.getrandbits(1)==0 else -0.1, 2) for _ in range(len(subject.descriptor))]
        super(Affinity, self).__init__("affinity", subject, t0prior)

class Bias(LatentVariable):
    def __init__(self, subject, t0prior=None):
        t0prior = t0prior if t0prior is not None else Gaussian(0, 0.5)
        super(Bias, self).__init__("bias", subject, t0prior)

class Thresholds(LatentVariable):
    def __init__(self, subject, thresholds=5, t0prior=None):
        t0prior = t0prior if t0prior if t0prior is not None else [Gaussian(0.1 if random.getrandbits(1)==0 else -0.1, 2) for _ in range(thresholds)]

class Subject:
    def __init__(self, id, type, descriptor, latentVariables=None, seenInSessions=None):
        self.initialize(id, type, descriptor, latentVariables, seenInSessions)
        
    def initialize(self, id, type, descriptor, latentVariables, seenInSessions):
        self.id = id
        self.type = type #TODO: not sure if this is necessary
        self.descriptor = descriptor
        if seenInSessions is None:
            seenInSessions = Timeline() #references to the sessions where there is a rating pertaining this subject
        self.seenInSessions = seenInSessions
        if latentVariables is None:
            latentVariables = [Affinity(self), Bias(self)]
        self.latentVariables = latentVariables
    
    def getLatentVariable(self, varName):
        for x in self.latentVariables:
            if x.name == varName:
                return x
        else:
            raise ValueError
    def __hash__(self):
        return hash((self.type, self.id))
    def __eq__(self, other):
        return self.type == other.type and self.id == other.id
    def __ne__(self, other):
        return not(self == other)
    def __repr__(self):
        return f"{self.__class__.__name__}({self.id},{self.descriptor})"

class User(Subject):
    _instance_counter = itertools.count(0)
    def __init__(self, descriptor):
        id = next(User._instance_counter)
        super(User, self).__init__(id, "user", descriptor)

class Movie(Subject):
    _instance_counter = itertools.count(0)
    def __init__(self, descriptor):
        id = next(Movie._instance_counter)
        super(Movie, self).__init__(id, "movie", descriptor)

class Rating:
    _instance_counter = itertools.count(0)

    def __init__(self, user, movie, val, comparisonMode=False, plotRatings=True, debugMode=True):
        self.id = next(Rating._instance_counter)
        self.user = user
        self.movie = movie
        self.subjects = {User: self.user, Movie: self.movie}
        self.likelihoodPerVariable = {}
        assert val is None or val >= 1 or val <= -1, f"Values with a small absolute value seem to "
        self.value = val
        self.calls_to_estimate = 0
        self.comparisonMode = comparisonMode
        self.plotRatings = plotRatings
        self.debugMode = debugMode
        self.estimate = Uniform()

    def __hash__(self):
        return hash(self.id)
    def __eq__(self, other):
        return self.id == other.id
    def __ne__(self, other):
        return not(self == other)
    def __repr__(self):
        return f"{self.value} (movie: {self.movie.id}, user: {self.user.id})"

    def updateVariableLikelihoods(self, session):
        modeIsRatingEstimation = self.value is None
        userDescriptorWP = session.withinPrior(self.user.getLatentVariable("affinity"))
        movieDescriptorWP = session.withinPrior(self.movie.getLatentVariable("affinity"))
        
        K = len(userDescriptorWP)
        assert K == len(movieDescriptorWP)

        userBiasWP = session.withinPrior(self.user.getLatentVariable("bias"))
        movieBiasWP = session.withinPrior(self.movie.getLatentVariable("bias"))
        bias = MessageFactory.msg1_bias(userBiasWP, movieBiasWP)

        msg1S = userDescriptorWP  #TODO: when we leave collab filtering we will need to do lineal combination of many wpriors (afaik)
        msg1T = movieDescriptorWP #TODO: related to above, will bias calculations change in the same manner?

        msg7S = [Uniform() for _ in range(K)] #dont have necessary data yet to compute msg7, so we initialize to Uniform
        msg7T = [Uniform() for _ in range(K)] #idem (Matchbox paper p.4 third-to-last paragraph)

        rating = Uniform() #remove if possible?
        
        # Cuando queremos comparar modelos queremos comparar las predicciones a priori de r.
        # En ese caso reemplazamos el msg6 por una uniforme porque al no observar el rating
        # el bias es independiente de los zs ie no afecta a z.

        # Revisar por que z y bias son independientes (concepto d-separation, en particular la estructura collider) 
        # cuando no observamos el rating. Tambien por que r no afecta a z cuando no es observado el r.

        estimationHasConverged = False
        iterations = 0
        ratings = []
        while not estimationHasConverged or iterations < 10:
            margS = np.multiply(msg1S, msg7S)
            margT = np.multiply(msg1T, msg7T)
            normZ = MessageFactory.msg2(margS, margT)
            
            newRating = MessageFactory.ratingEstimate(bias, normZ)
            if modeIsRatingEstimation:
                # Estimate unknown rating
                norm6 = [Uniform() for _ in range(K)]
            else:
                # Process new rating
                if self.plotRatings:
                    plotMsg2(normZ, self.value, margS, margT, f"{self.id}_{self.calls_to_estimate}", iterations, makeDir=True)
                norm6 = MessageFactory.msg6(self.value, bias, normZ, session.betaNoise2)
            msg7S = MessageFactory.msg7(norm6, margT)
            msg7T = MessageFactory.msg7(norm6, margS)
            
            if not modeIsRatingEstimation and self.plotRatings:
                plotMsg7(msg7S, norm6, margT, self.value, f"{self.id}_{self.calls_to_estimate}", iterations, makeDir=True)
            
            if iterations == 0:
                aprox0 = normZ[0].eval(np.arange(X_LEFT_BOUND,X_RIGHT_BOUND,X_STEP))
            estimationHasConverged = MessageFactory.estimationsConverge(rating, newRating)
            rating = newRating
            ratings.append(rating)
            iterations += 1
        
        if not modeIsRatingEstimation and self.debugMode:
            exact7n = MessageFactory.exactmsg7(norm6, margT, np.arange(X_LEFT_BOUND,X_RIGHT_BOUND,X_STEP), np.arange(X_LEFT_BOUND,X_RIGHT_BOUND,X_STEP))
            #print(f"[var_{self.id}_{self.calls_to_estimate}] KL divergence between aprox msg 7 in it 0 and exact msg 7 in it {iterations}: {MessageFactory.KLdivergence(aprox0, exact7n, 0.1)}")     
        
        if not modeIsRatingEstimation and self.plotRatings:
            #plotRatings(ratings, self.value, f"{self.id}_{self.calls_to_estimate}")
            dir = f"{IMG_DIR}/msg7/var_{self.id}_{self.calls_to_estimate}/"
            for descriptor in os.listdir(dir):
                imageio.mimsave(f'{dir}{descriptor}/{descriptor}.gif', [imageio.imread(dir+descriptor+"/"+pname) for pname in os.listdir(dir+descriptor+"/")], fps=2)
                print(f'{dir}{descriptor}.gif')

        # since lhood_b doesnt affect the other estimates but is affected by them we
        # #calcular# them after convergence
        if not modeIsRatingEstimation:
            lhood_b = MessageFactory.lhood_bias(self.value, normZ, session.betaNoise2) #TODO: esto retroalimenta la estimacion de bias? en user y movie se retroalimentaba por las estimaciones creo, aca lo que ajusta seria normz?
            lhood_bUser = MessageFactory.lhood_subjectBias(lhood_b, movieBiasWP)
            lhood_bMovie = MessageFactory.lhood_subjectBias(lhood_b, userBiasWP)
        else: #TODO: might be better off not updating or not using the dummy rating for estimates
            lhood_b = MessageFactory.lhood_bias(rating.mu, normZ, session.betaNoise2) #TODO: esto retroalimenta la estimacion de bias? en user y movie se retroalimentaba por las estimaciones creo, aca lo que ajusta seria normz?
            lhood_bUser = MessageFactory.lhood_subjectBias(lhood_b, movieBiasWP)
            lhood_bMovie = MessageFactory.lhood_subjectBias(lhood_b, userBiasWP)

        self.likelihoodPerVariable[self.user.getLatentVariable("affinity")] = msg7S
        self.likelihoodPerVariable[self.movie.getLatentVariable("affinity")] = msg7T
        self.likelihoodPerVariable[self.user.getLatentVariable("bias")] = lhood_bUser
        self.likelihoodPerVariable[self.movie.getLatentVariable("bias")] = lhood_bMovie
        self.estimate = rating
        self.calls_to_estimate += 1

def plotRatings(ratings, val, name, makeDir=True):
    tmp = np.arange(X_LEFT_BOUND,X_RIGHT_BOUND,X_STEP)
    fig = plt.figure()
    for it in range(len(ratings)):
        plt.title(f"Iteración {it}")
        plt.plot(tmp, ratings[it].eval(tmp), label="estimation")
        plt.axvline(x=val, linestyle="dashed", color="black")
        plt.legend()
        path = f'./rating/{name}'
        if makeDir:
            if not os.path.exists(path):
                os.makedirs(path)
            fname = path+f"/{it}.png"
        else:
            fname = path+f'_{it}.png'
        plt.savefig(fname)
        plt.close()
        plt.clf()

def plotMsg2(normZ, val, margS, margT, name, iteration, makeDir=True):
    step = 0.1
    zk = np.arange(X_LEFT_BOUND,X_RIGHT_BOUND,X_STEP)
    fig = plt.figure()
    exact = MessageFactory.exactmsg2(margS, margT, zk)

    for it in range(len(normZ)):
        aprox = normZ[it].eval(zk)
        kl_divergence = MessageFactory.KLdivergence(aprox, exact, step)
        plt.title(f"Iteración {iteration} (divergencia {kl_divergence})")
        plt.plot(zk, aprox, label="aproximado")
        plt.plot(zk, exact, label="exacto")
        plt.axvline(x=val, linestyle="dashed", color="black")
        plt.legend()
        path = f'{IMG_DIR}/msg2/var_{name}/desc_{it}/'
        if makeDir:
            if not os.path.exists(path):
                os.makedirs(path)
            fname = path+f"/k0-{iteration}.png"
        else:
            fname = path+f'_k0-{iteration}.png'
        plt.savefig(fname)
        plt.clf()

def plotMsg7(norm7, norm6, margT, val, name, iteration, makeDir=True):
    step = 0.1
    sk = np.arange(X_LEFT_BOUND,X_RIGHT_BOUND,X_STEP)
    tk = sk
    exact = MessageFactory.exactmsg7(norm6, margT, sk, tk)

    for it in [0]:#range(len(norm7)):
        aprox = norm7[it].eval(sk)
        kl_divergence = MessageFactory.KLdivergence(aprox, exact, step)
        #bestRes = minimize(lambda x: MessageFactory.KLdivergence(Gaussian(x[0], x[1]).eval(sk), exact, step), [2,0.2])
        #assert bestRes.success, f"Msg7 optimization failed with message {bestRes.message}"
        #best = Gaussian(bestRes.x[0],bestRes.x[1]).eval(sk)
        plt.title(f"Iteración {iteration} (divergencia {kl_divergence})")
        plt.plot(sk, aprox, label="aproximado")
        plt.plot(sk, exact, label="exacto")
        #plt.plot(sk, best, label="best")
        plt.axvline(x=val, linestyle="dashed", color="black")
        plt.legend()
        path = f'{IMG_DIR}/msg7/var_{name}/desc_{it}/'
        if makeDir:
            if not os.path.exists(path):
                os.makedirs(path)
            fname = path+f"/k0-{iteration}.png"
        else:
            fname = path+f'_k0-{iteration}.png'
        plt.savefig(fname)
        plt.clf()

class Session:
    def __init__(self, time, history, betaNoise2=1, posteriorsPerVariable=None, ratings=None):
        self.time = time
        self.history = history
        self.betaNoise2 = betaNoise2
        #A dict where keys are latent variables and values are dict with "forward" and "backward" messages
        #(check Session#updatePosteriors)
        self.posteriorsPerVariable = {} if posteriorsPerVariable is None else posteriorsPerVariable
        self.ratings = set() if ratings is None else ratings
    
    def addRating(self, r):
        r.updateVariableLikelihoods(self)
        self.ratings.add(r)
        # Update posterior messages for the latent variables of the subjects in the rating
        for latentVar in r.likelihoodPerVariable.keys():
            self.updatePosteriors(latentVar) #New forward takes into account new likelihood because we added it to the ratings list
    
    def updatePosteriors(self, latentVar):
        subjectLhoods = self.getVariableLikelihoods(latentVar)
        fp = self.history.forwardPrior(latentVar, self.time)
        bp = self.history.backwardPrior(latentVar, self.time)
        try:
            self.posteriorsPerVariable[latentVar]["forward"] = np.prod([bp]+subjectLhoods, axis=0)
        except KeyError:
            self.posteriorsPerVariable[latentVar] = {}
            self.posteriorsPerVariable[latentVar]["forward"] = np.prod([bp]+subjectLhoods, axis=0)
        self.posteriorsPerVariable[latentVar]["backward"] = np.prod([fp]+subjectLhoods, axis=0)

    def withinPrior(self, latentVar):
        lhoods = self.getVariableLikelihoods(latentVar)
        fp = self.history.forwardPrior(latentVar, self.time)
        bp = self.history.backwardPrior(latentVar, self.time) #TODO: chequear que anda bien para estimaciones no arrays
        return np.prod([fp, bp] + lhoods, axis=0)

    def getVariableLikelihoods(self, latentVar):
        return [r.likelihoodPerVariable[latentVar] for r in self.ratings if latentVar in r.likelihoodPerVariable]

    def updateVariableLikelihoods(self):
        for r in self.ratings:
            r.updateVariableLikelihoods(self)

    def updateAllPosteriors(self):
        for latentVar in self.posteriorsPerVariable.keys():
            self.updatePosteriors(latentVar)

    def reprocessSession(self):
        self.updateVariableLikelihoods()
        self.updateAllPosteriors()

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.time})"

class History:
    def __init__(self, mu0, sigma0, gammaNoise2):
        self.allSubjects = {User:set(), Movie:set()}
        self.sessions = Timeline()
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.gammaNoise2 = gammaNoise2

    def addRating(self, r, t):
        try:
            session = self.sessions.getEventByTime(t)
        except ValueError:
            # Rating belongs to new session
            session = Session(t, self)
            self.sessions.place(session)
        
        session.addRating(r)
        # Keep track of what sessions we saw the user and movie in
        r.subjects[User].seenInSessions.ifAbsentPlace(session)
        r.subjects[Movie].seenInSessions.ifAbsentPlace(session)
        self.allSubjects[User].add(r.subjects[User])
        self.allSubjects[Movie].add(r.subjects[Movie])

    def dynamicUncertainty(self, prior, timeDifference):
        try:
            iter(prior)
            return [Gaussian(p.mu, np.sqrt(p.sigma2 + self.gammaNoise2*timeDifference)) for p in prior]
        except TypeError:
            #TODO: timeDifference se multiplica antes del cuadrado de gamma? Revisar matematica
            return Gaussian(prior.mu, np.sqrt(prior.sigma2 + self.gammaNoise2*timeDifference))

    # return forward prior for given subject on time t 
    # (Uniform if subject does not appear before t, corresponding posterior with noise otherwise)
    def forwardPrior(self, latentVar, time):
        try:
            session = latentVar.subject.seenInSessions.eventBefore(time)
            fp = session.posteriorsPerVariable[latentVar]["forward"]
            return self.dynamicUncertainty(fp, time - session.time)
        except ValueError:
            #Edge case: if no forwardPrior should be prior0!!
            return latentVar.t0prior

    # return backward prior for given subject on time t 
    # (Uniform if subject does not appear after t, corresponding posterior with noise otherwise)
    def backwardPrior(self, latentVar, time):
        try:
            session = latentVar.subject.seenInSessions.eventAfter(time)
            bp = session.posteriorsPerVariable[latentVar]["backward"]
            return self.dynamicUncertainty(bp, session.time - time)
        except ValueError:
            try:
                return [Uniform() for _ in latentVar.t0prior]
            except TypeError:
                return Uniform()

    #TODO: revisar si la propagacion ya al agregar ratings se hace sola o si hay casos no cubiertos eg insercion no cronologica
    def propagate(self):
        estimationsConverge = False
        while (not estimationsConverge):
            estimationsConverge = True
            posteriorsBefore = {s:s.posteriorsPerVariable.copy() for s in self.sessions}
            for s in self.sessions: #propagate messages forward
                s.reprocessSession()

            for s in reversed(self.sessions): #propagate messages backward
                s.reprocessSession()
                # will be true if all estimations converge:
                for latentVar, estim in posteriorsBefore[s].items():
                    if not estimationsConverge:
                        break
                    new = s.posteriorsPerVariable[latentVar]
                    
                    try:
                        for i in range(len(new["forward"])):
                            estimationsConverge = estimationsConverge and MessageFactory.estimationsConverge(estim["forward"][i], new["forward"][i])
                        for i in range(len(new["backward"])):
                            estimationsConverge = estimationsConverge and MessageFactory.estimationsConverge(estim["backward"][i], new["backward"][i])
                    except TypeError:
                        estimationsConverge = estimationsConverge and MessageFactory.estimationsConverge(estim["forward"], new["forward"])
                        estimationsConverge = estimationsConverge and MessageFactory.estimationsConverge(estim["backward"], new["backward"])
                    
                
    #TODO: add ability to predict ratings for sessions that dont exist (pass session as an integer)           
    #TODO: the thing before was added in a rush and might not be working properly
    def estimateRating(self, user, movie, sessionT=None):
        r = Rating(user, movie, val=None)
        try:
            sessionT = Session(len(self.sessions), self) if sessionT is None else self.sessions.getEventByTime(sessionT)
        except ValueError:
            sessionT = Session(sessionT, self)
        r.updateVariableLikelihoods(sessionT)
        return r.estimate

class MessageFactory:
    @staticmethod
    def msg1_bias(ubias, vbias):
        return Gaussian(ubias.mu+vbias.mu, np.sqrt(ubias.sigma2 + vbias.sigma2))

    @staticmethod
    def msg1(mat, v):
        return mat@v

    @staticmethod
    def msg2(margS, margT):
        res = []
        for k in range(len(margS)):
            s = margS[k]
            t = margT[k]
            res.append(Gaussian(s.mu*t.mu, (s.ncsm*t.ncsm) - ((s.mu**2)*(t.mu**2))))
        return res

    @staticmethod
    def lhood_bias(r, normZ, betaNoise2):
        sumMuZ = sum([zk.mu for zk in normZ])
        sumSigma2Z = sum([zk.sigma2 for zk in normZ])
        return Gaussian(r - sumMuZ, np.sqrt(betaNoise2 + sumSigma2Z))

    @staticmethod
    def lhood_subjectBias(normB, normVbias):
        return Gaussian(normB.mu-normVbias.mu, np.sqrt(normB.sigma2 + normVbias.sigma2))

    @staticmethod
    def ratingEstimate(b, normZ):
        sumMuZ = sum([zk.mu for zk in normZ])
        sumSigma2Z = sum([zk.sigma2 for zk in normZ])
        return Gaussian(b.mu + sumMuZ, np.sqrt(b.sigma2 + sumSigma2Z))

    @staticmethod
    def msg6(r, b, normZ, betaNoise2):
        res = []
        for k in range(len(normZ)):
            mu6 = r - (b.mu + sum([normZ[i].mu for i in range(len(normZ)) if i != k]))
            sigma26 = betaNoise2 + b.sigma2 + sum([normZ[i].sigma2 for i in range(len(normZ)) if i != k]) #since zk is who the msg is sent to, normZ[k] is not present in the message to sent to it (doesn't participate in estimating itself?)
            res.append(Gaussian(mu6, np.sqrt(sigma26)))
        return res

    @staticmethod
    def msg7(norm6, margT):
        assert len(norm6) == len(margT)
        res = []
        for k in range(len(margT)):
            m6 = norm6[k]
            t = margT[k]
            #nuestra varianza: m6.variance/(t.mu**2+t.variance)
            #precision dotnet: (t.variance+t.mu**2)/m6.variance
            # => nuestra varianza es equivalente a la precision de dotnet
            # PERO salva los casos en que alguna de las dos sea point mass (???)
            #
            # Como nota, no parece estar guardando la media? Solo meanTimesPrecision:
            # meanTimesPrecisionDotnet: (m6.mu/m6.variance) * t.mu = m7.mu/m7.variance
            # (m6.mu/m6.variance) * t.mu * m7.variance = m7.mu
            # (m6.mu/m6.variance) * t.mu * m6.variance/(t.mu**2+t.variance)
            # m6.mu*t.mu / t.mu+t.variance

            # pruebo poner eso como mu pero probablemente sea incorrecto.
            # original: 
            res.append(Gaussian((m6.mu*t.mu)/t.ncsm, np.sqrt((m6.ncsm-m6.mu**2)/t.ncsm)))
        return res

    @staticmethod
    def posterior(wprior, likelihoods):
        res = wprior
        for l in likelihoods:
            res = np.multiply(res, l)
        return res

    @staticmethod
    def estimationsConverge(estimation, newEstimation):
        muConverge = (estimation.mu - newEstimation.mu) < EPSILON
        sigmaConverge = (estimation.sigma - newEstimation.sigma) < EPSILON
        return muConverge and sigmaConverge

    @staticmethod
    def KLdivergence(x, y, step):
        #assert 0 not in x, "0 will appear in denominator"
        #res = sum([x[i]*np.log(x[i]/y[i])*step for i in range(len(x))])
        #rel_entr(exact, aprox)res = sum([-aprox[i]*np.log(exact[i]/aprox[i])*step for i in range(len(aprox))])
        
        #assert (sum(rel_entr(x, y)*step) == res).all(), "Calculated KL divergence was different from scipy kl div"
        res = sum(rel_entr(np.array(x)*step, np.array(y)*step))
        return res
    
    @staticmethod
    def exactmsg7(norm6, margT, sk, tk):
        exact = []
        for s in sk:
            #probamos agregar logaritmo siguiendo lo que se ve en la documentacion de AAverageLogarithm, que es donde esta implementado
            # el mensaje aproximado 7 de infer.NET. Aca: https://dotnet.github.io/infer/apiguide/api/Microsoft.ML.Probabilistic.Factors.GaussianProductVmpOp.html
            exact.append(sum(np.log(norm6[0].eval([s*t for t in tk]))*margT[0].eval(tk)))
        #exact = [e/(sum(exact)*0.1) for e in exact]
        return exact
    
    @staticmethod
    def exactmsg2(margS, margT, zk):
        exact = []
        for x in zk:
            exact.append(sum(margS[0].eval(x/zk)*margT[0].eval(zk)))
        exact = exact/(sum(exact)*0.1)
        return exact
