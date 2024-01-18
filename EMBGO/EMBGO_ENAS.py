
from copy import deepcopy
import numpy as np
import warnings
from ENAS_Data.robustness_dataset import RobustnessDataset
import os
from scipy.stats import levy
warnings.filterwarnings("ignore")


Data = RobustnessDataset(path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\ENAS_Data")
DataName = "cifar10"
Metric = "clean"  # candicates = ["clean", "aa_apgd-ce@Linf", "aa_square@Linf", "fgsm@Linf", "pgd@Linf"]
Results = Data.query(
    data=[DataName],
    measure="accuracy",
    key=RobustnessDataset.keys_clean + RobustnessDataset.keys_adv + RobustnessDataset.keys_cc
)


def truncation(indi):
    tmpIndi = np.zeros(len(indi))
    for i in range(len(indi)):
        if indi[i] < -60:
            tmpIndi[i] = 0
        elif indi[i] < -20:
            tmpIndi[i] = 1
        elif indi[i] < 20:
            tmpIndi[i] = 2
        elif indi[i] < 60:
            tmpIndi[i] = 3
        else:
            tmpIndi[i] = 4
    return tmpIndi

def transfer(tmpIndi):
    uid = 0
    for i in range(len(tmpIndi)):
        uid += tmpIndi[i] * 5 ** i
    return int(uid)


def fit_func(indi):
    global Metric, Results, DataName
    tmpIndi = truncation(indi)
    uid = transfer(tmpIndi)
    if Metric == "clean":
        acc = Results[DataName][Metric]["accuracy"][Data.get_uid(uid)]
    else:
        acc = Results[DataName][Metric]["accuracy"][Data.get_uid(uid)][Data.meta["epsilons"][Metric].index(1.0)]
    return acc


PopSize = 50  # the number of Pop
DimSize = 6  # the number of variables
LB = [-100] * DimSize  # the maximum value of the variable range
UB = [100] * DimSize  # the minimum value of the variable range
TrialRuns = 30  # the number of independent runs
MaxFEs = 5000  # the maximum number of fitness evaluations

Pop = np.zeros((PopSize, DimSize))  # the coordinates of the individual (candidate solutions)
FitPop = np.zeros(PopSize)  # the fitness value of all Pop
curFEs = 0  # the current number of fitness evaluations
FuncNum = 1  # the serial number of benchmark function
curIter = 0  # the current number of generations
MaxIter = int(MaxFEs / PopSize)
curBest = np.zeros(DimSize)  # the best individual in the current generation
FitBest = 0  # the fitness of the best individual in the current generation
curWorst = np.zeros(DimSize)  # the worst individual in the current generation


# initialize the M randomly
def Initialization(func):
    global Pop, FitPop, curBest, FitBest
    # randomly generate Pop
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitPop[i] = -func(Pop[i])
    bestIdx = np.argmin(FitPop)
    curBest = Pop[bestIdx].copy()
    FitBest = FitPop[bestIdx]


def Check(indi):
    global LB, UB
    for i in range(len(indi)):
        if indi[i] > UB[i] or indi[i] < LB[i]:
            indi[i] = np.random.uniform(LB[i], UB[i])
    return indi


def EMBGO(func):
    global Pop, FitPop, curBest, FitBest, curWorst, curIter, MaxIter
    # update the current worst Pop
    worstIdx = np.argmax(FitPop)
    curWorst = Pop[worstIdx].copy()
    # calculate the distance between the best individual and the worst individual
    maxDist = np.linalg.norm(curBest - curWorst + np.finfo(float).eps, ord=2)
    safeRadius = np.random.uniform(0.8, 1.2) * maxDist
    Xmean = np.mean(Pop, axis=0)
    # record the generated Off individual
    Off = np.zeros(DimSize)
    # movement of Pop in the first round
    for i in range(PopSize):
        if np.random.rand() < 0.5:
            # calculate the distance between the current individual and the best Pop
            distance = np.linalg.norm(curBest - Pop[i] + np.finfo(float).eps, ord=2)
            # if the individual falls within the range of the first round, it indicates that the individual has potential.
            if distance < safeRadius:
                Off = Pop[i] + np.sin(np.random.rand() * 2 * np.pi) * (curBest - Pop[i]) + np.sin(
                    np.random.rand() * 2 * np.pi) * (Xmean - Pop[i])
            else:
                for j in range(DimSize):
                    Off[j] = Pop[i][j] + levy.rvs()
        else:
            # battle Phase
            selectedIdx = np.random.randint(0, PopSize)
            while selectedIdx == i:
                selectedIdx = np.random.randint(0, PopSize)
            # compare the fitness of two Pop, if the randomly selected individual is better
            if FitPop[i] > FitPop[selectedIdx]:
                # compute the vector between two Pop
                space = Pop[selectedIdx] - Pop[i]
                for j in range(DimSize):
                    if np.random.uniform() < 0.5:
                        Off[j] = Pop[i][j] + space[j] * 0.5 * np.random.rand()
                    else:
                        Off[j] = Pop[selectedIdx][j] + space[j] * 0.5 * np.random.rand()
            else:
                # compute the vector between two Pop
                space = Pop[i] - Pop[selectedIdx]
                Off = Pop[i] + space * np.cos(np.random.rand() * 2 * np.pi)

        Off = Check(Off)
        FitOff = -func(Off)
        # If the Off individual is better, replace its parent.
        if FitOff < FitPop[i]:
            Pop[i] = Off.copy()
            FitPop[i] = FitOff
            if FitOff < FitBest:
                curBest = Off.copy()
                FitBest = FitOff


def RunEMBGO(func):
    global curFEs, curIter, MaxFEs, TrialRuns, DimSize, Metric
    All_Trial_Best = []
    for i in range(TrialRuns):
        BestList = []
        curFEs = 0
        curIter = 0
        Initialization(func)
        BestList.append(FitBest)
        np.random.seed(2024 + 88 * i)
        while curIter <= MaxIter:
            EMBGO(func)
            curIter += 1
            BestList.append(FitBest)
        All_Trial_Best.append(np.abs(BestList))
    np.savetxt("./EMBGO_Data/ENAS/" + DataName + "_" + str(Metric) + ".csv", All_Trial_Best, delimiter=",")


def main(dataname):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB, Results, DataName, Metric
    DataName = dataname
    DimSize = 6
    Pop = np.zeros((PopSize, DimSize))
    MaxFEs = 5000
    MaxIter = int(MaxFEs / PopSize)

    Results = Data.query(
        data=[dataname],
        measure="accuracy",
        key=RobustnessDataset.keys_clean + RobustnessDataset.keys_adv + RobustnessDataset.keys_cc
    )

    Indicators = ["clean", "aa_apgd-ce@Linf", "aa_square@Linf", "fgsm@Linf", "pgd@Linf"]

    FuncNum = 0
    for i in range(len(Indicators)):
        Metric = Indicators[i]
        RunEMBGO(fit_func)


if __name__ == "__main__":
    if os.path.exists('./EMBGO_Data/ENAS') == False:
        os.makedirs('./EMBGO_Data/ENAS')
    Datasets = ["cifar10", "cifar100"]
    for data in Datasets:
        main(data)


