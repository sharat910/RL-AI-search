from learn import MarkovAgent
from search import *
import numpy as np

simulator = SearchSimulation()
observations = simulator.observations(50000, 15)
mark = MarkovAgent(observations)
mark.learn()


class AISearch(Search):
  def update_location(self):
    self.location = mark.policy[self.state()]

binary_results = []
linear_results = []
random_results = []
ai_results = []

for i in xrange(10000):
  # create array and target value
  array = simulator._random_sorted_array(15)
  target = random.choice(array)

  # generate observation for search of each type
  binary = simulator.observation(len(array),BinarySearch(array, target))
  linear = simulator.observation(len(array),LinearSearch(array, target))
  rando = simulator.observation(len(array),RandomSearch(array, target))
  ai = simulator.observation(len(array),AISearch(array, target))

  # append result
  binary_results.append(len(binary['state_transitions']))
  linear_results.append(len(linear['state_transitions']))
  random_results.append(len(rando['state_transitions']))
  ai_results.append(len(ai['state_transitions']))

# display average results
print "Average binary search length: {0}".format(np.mean(binary_results)) # 3.6469
print "Average linear search length: {0}".format(np.mean(linear_results)) # 5.5242
print "Average random search length: {0}".format(np.mean(random_results)) # 14.2132
print "Average AI search length: {0}".format(np.mean(ai_results)) # 3.1095