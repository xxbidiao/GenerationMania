# DifficultyCalculation.py
# Utility functions for difficulty calculations.
# osu difficulty calculator adapted from:
# https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/Difficulty/ManiaDifficultyCalculator.cs
import numpy as np
import matplotlib.pyplot as plt
from .data_utils import translateColumnNameToNumber

def find_timed_result(data,time):
    return data.get(time, data[min(data.keys(), key=lambda k: k - time if k>=time-(1e-8) else float('inf'))])

# Normalization factors for each difficulty calculator.
NAIVE_NORMALIZATION_FACTOR = 0.3

# Internal const variables inside ManiaHitObjectDifficulty
INDIVIDUAL_DECAY_BASE = 0.125
OVERALL_DECAY_BASE = 0.30

#static definition for number of columns
columnCount = 8

# This is a direct translation for osu!mania's star difficulty calculation.
# See their code base for more details.
class ManiaHitObjectDifficulty():
    def __init__(self, baseHitObject, columnCount):
        self.BaseHitObject = baseHitObject

        if (self.BaseHitObject['end'] == None):
            self.BaseHitObject['end'] = self.BaseHitObject['time']

        self.startTime = self.BaseHitObject['time']
        self.endTime = self.BaseHitObject['end']

        self.beatmapColumnCount = columnCount
        self.heldUntil = [0 for _ in range(self.beatmapColumnCount)]
        self.individualStrains = [0 for _ in range(self.beatmapColumnCount)]

        self.OverallStrain = 1

    def calculateStrains(self, previousHitObject, timeRate):
        timeElapsed = (self.BaseHitObject['time'] - previousHitObject.startTime) / timeRate
        individualDecay = INDIVIDUAL_DECAY_BASE ** timeElapsed
        overallDecay = OVERALL_DECAY_BASE ** timeElapsed

        holdFactor = 1.0 # Factor to all additional strains in case something else is held
        holdAddition = 0 # Addition to the current note in case it's a hold and has to be released awkwardly

        # Fill up the heldUntil array
        for i in range(self.beatmapColumnCount):
            self.heldUntil[i] = previousHitObject.heldUntil[i]

            # If there is at least one other overlapping end or note, then we get an addition, buuuuuut...
            if (self.BaseHitObject['time'] < self.heldUntil[i] and self.endTime > self.heldUntil[i]):
                holdAddition = 1.0

            # ... this addition only is valid if there is _no_ other note with the same ending. Releasing multiple notes at the same time is just as easy as releasing 1
            if (self.endTime == self.heldUntil[i]):
                holdAddition = 0

            # We give a slight bonus to everything if something is held meanwhile
            if (self.heldUntil[i] > self.endTime):
                holdFactor = 1.25

            # Decay individual strains
            self.individualStrains[i] = previousHitObject.individualStrains[i] * individualDecay

        self.heldUntil[translateColumnNameToNumber(self.BaseHitObject['data']['column'])] = self.endTime  #TODO is this the correct implementation?

        # Increase individual strain in own column
        self.individualStrains[translateColumnNameToNumber(self.BaseHitObject['data']['column'])] += 2.0 * holdFactor

        self.OverallStrain = previousHitObject.OverallStrain * overallDecay + (1.0 + holdAddition) * holdFactor

    def IndividualStrain(self):
        return self.individualStrains[translateColumnNameToNumber(self.BaseHitObject['data']['column'])]


class DifficultyCalculator():
    def __init__(self,method="naive"):
        self.lastOsuStrain = None
        self.lastDifficulty = 0
        self.method = method
        self.star_scaling_factor = 0.018
        self.strain_step = 0.4
        self.decay_weight = 0.9
        self.difficultyHitObjects = list()
        self.TimeRate = 1
        self.starDifficultyCap = 999.9

    def calculate(self, chart):
        #print("Total notes:%d Method:%s"%(len(chart['playables']),self.method))
        if self.method == "naive":
            self.lastDifficulty = self.calculateNaiveDifficulty(chart)
        elif self.method == "osu":
            self.lastDifficulty = self.calculateOSUDifficulty(chart)
        else:
            self.lastDifficulty = -1
            raise TypeError("Unsupported Difficulty calculation method specified.")
        return self.lastDifficulty

    # This method simply returns normalized per-second density.
    def calculateNaiveDifficulty(self, chart):
        times = []
        for playables in chart['playables']:
            times.append(playables['time'])
        if len(times) < 1:
            return 0
        result = len(times) / (max(times) - min(times))
        result *= NAIVE_NORMALIZATION_FACTOR
        return result

    def calculateOSUDifficulty(self, chart, categoryDifficulty = None):
        # Fill our custom DifficultyHitObject class, that carries additional information
        self.difficultyHitObjects = list()

        #columnCount = 7  Note that this is subject to change; Currently defined statically

        for hitObject in chart['playables']:  #TODO is this the correct implmentation?
            self.difficultyHitObjects.append(ManiaHitObjectDifficulty(hitObject, columnCount))

        self.difficultyHitObjects.sort(key=lambda a: a.startTime)  #osu implmentation had "StartTime"
        #print("DifficultyHO count:%d"%len(self.difficultyHitObjects))
        if not self.calculateStrainValues():
            return 0

        starRating = self.calculateDifficulty() * self.star_scaling_factor

        if categoryDifficulty != None:
            categoryDifficulty['strain'] = starRating

        return min(starRating, self.starDifficultyCap)

    def calculateStrainValues(self):
        if (len(self.difficultyHitObjects) < 2):
            return False

        current = self.difficultyHitObjects[0]

        i = 1
        while i < len(self.difficultyHitObjects):
            nextHitObject = self.difficultyHitObjects[i]
            if (nextHitObject != None):
                nextHitObject.calculateStrains(current, self.TimeRate)
            current = nextHitObject
            i += 1

        return True

    def getOsuNoteStrain(self,time):
        try:
            return find_timed_result(self.lastOsuStrain,time)
        except:
            print("Trying to get OsuNoteStrain without calculating osu strain.")
            return -1
    def calculateDifficulty(self):
        actualStrainStep = self.strain_step * self.TimeRate

        # Find the highest strain value within each strain step
        highestStrains = list()
        strain_with_time = {}
        intervalEndTime = actualStrainStep
        maximumStrain = 0 # We need to keep track of the maximum strain in the current interval

        previousHitObject = None # TODO make typedef for ManiaHitObjectDifficulty

        for hitObject in self.difficultyHitObjects:
            # While we are beyond the current interval push the currently available maximum to our strain list
            while (hitObject.startTime > intervalEndTime):
                highestStrains.append(maximumStrain)
                strain_with_time[intervalEndTime] = maximumStrain
                # The maximum strain of the next interval is not zero by default! We need to take the last hitObject we encountered, take its strain and apply the decay
                # until the beginning of the next interval.
                if (previousHitObject == None):
                    maximumStrain = 0
                else:
                    individualDecay = INDIVIDUAL_DECAY_BASE ** (intervalEndTime - previousHitObject.startTime)
                    overallDecay = OVERALL_DECAY_BASE ** (intervalEndTime - previousHitObject.startTime)
                    maximumStrain = previousHitObject.IndividualStrain() * individualDecay + previousHitObject.OverallStrain * overallDecay

                intervalEndTime += actualStrainStep

            strain = hitObject.IndividualStrain() + hitObject.OverallStrain #TODO figure out strain values
            maximumStrain = max(strain, maximumStrain)

            previousHitObject = hitObject

        difficulty = 0
        weight = 1

        self.lastOsuStrain = strain_with_time


        #print("Strain #:%d"%len(highestStrains))
        #h2 = highestStrains[:]

        highestStrains.sort(reverse=True) # Sort from highest to lowest strain.

        for strain in highestStrains:
            difficulty += weight * strain
            weight *= self.decay_weight


        def moving_average(a, n=20):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n
        # This shows a figure of the difficulty curve of each chart processed.
        #perNoteStrain = moving_average(h2)
        # perNoteStrain = strain_with_time
        # p_lists = sorted(perNoteStrain.items())  # sorted by key, return a list of tuples
        # x, y = zip(*p_lists)  # unpack a list of pairs into two tuples
        # plt.plot(x,y)
        # plt.xlabel('Time')
        # plt.ylabel('Difficulty')
        # plt.show()

        return difficulty
