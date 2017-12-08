import random
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import math

# Used to represent the Sinks on our board
class Sink:
    # Each sink is initialized with an x y coordinate
    def __init__(self, x, y):
        self.location = [x, y]

    # Setter method for the level of the tree this sink is on.
    def setLevel(self, level):
        self.level = level

    # Getter method for the level of the tree this sink is on.
    def getLevel(self):
        return self.level

class Centroid:

    # Every Centroid is initialized with an x y coordinate.
    def __init__(self, x, y):
        self.location = [x, y]

    # Number of sinks is determined by which level of the tree we're on.
    # Since we're only operating with one level for now, this isn't used.
    def set_number_of_sinks(self, n):
        self.num_sinks = n

# Represents the traits of our plot.
# Each element in the arrays correspond to a SinkGroup.
class PlotAttrs:
    def __init__(self):
        # matplot markers: https://matplotlib.org/api/markers_api.html
        self.centroid_attrs = ['rv', 'bv', 'gv', 'kv', 'mv']
        self.sink_attrs = ['ro', 'bo', 'go', 'ko', 'mo']
        self.line_colors = ['r', 'b', 'g', 'k', 'm']

# This class represents our SymmetricClockTree, and holds the majority of our logic.
class SymmetricClockTree:

    # Each tree is initalized with the number of sinks, and the number of levels
    # Can't handle multiple levels just yet, so we default to one
    def __init__(self, num_sinks, num_levels=1):
        self.num_sinks = num_sinks
        self.sinks = []
        self.centroids = []
        self.sinkGroups = {}
        self.minXCoordinate = 0
        self.maxXCoordinate = 5000
        self.minYCoordinate = 0
        self.maxYCoordinate = 5000
        self.standardizedWireLength = 0
        self.directionTraveled = { "up": False, "down": False, "left": False, "right": False }

    # Helper method for generating random sink locations.
    def generateRandomSinkLocations(self):
        radius = 500
        rangeX = (self.maxXCoordinate/4, (3*self.maxXCoordinate/4))
        rangeY = (self.maxYCoordinate/4, (3*self.maxYCoordinate/4))
        qty = self.num_sinks  # or however many points you want

        # Generate a set of all points within 200 of the origin, to be used as offsets later
        # There's probably a more efficient way to do this.
        deltas = set()
        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                if x*x + y*y <= radius*radius and x*x <= self.maxXCoordinate and y*y <= self.maxYCoordinate:
                    deltas.add((x,y))

        excluded = set()
        i = 0
        while i<qty:
            x = random.randrange(*rangeX)
            y = random.randrange(*rangeY)
            if (x,y) in excluded: continue
            self.sinks.append(Sink(x, y))
            i += 1
            excluded.update((x+dx, y+dy) for (dx,dy) in deltas)

    # We perform KMeans cluster to group our sinks into pre-defined cluster sizes.
    # TODO: Make the cluster sizes dynamic (based on number of sinks per grouping)
    def groupSinks(self):
        # Get all sinks locations
        sinks = self.getSinkLocations()
        # Perform KMeans clustering and identify cluster centers.
        kmeans = KMeans(n_clusters=3).fit(sinks)
        centroids = kmeans.cluster_centers_

        # For each identified cluster center:
        # Create a new Centroid
        # Add Centroid to our tree
        # Create a new SinkGroup
        for i, location in enumerate(centroids):
            centroid = Centroid(location[0], location[1])
            self.addCentroid(centroid, i)
            self.createSinkGroup(centroid, i)

        # Once we've made a SinkGroup for each cluster center
        # We'll use the list of groupings, and add each Sink to their designated group.
        self.addSinksToGroups(kmeans.labels_)

    # Returns the location of each of the sinks in our clock tree.
    def getSinkLocations(self):
        locations = []
        for sink in self.sinks:
            locations.append(sink.location)
        return locations

    # Adds a Centroid to our tree
    def addCentroid(self, centroid, i):
        self.centroids.append(centroid)

    # Creates a dictionary entry, which represents our SinkGroups.
    # SinkGroups are indexed by their group_id (i)
    def createSinkGroup(self, centroid, i):
        sinkGroupDict = {
            "centroid_location": [centroid.location[0], centroid.location[1]],
            "sinks": []
        }

        self.sinkGroups[i] = sinkGroupDict

    # Adds each sink to their designated SinkGroup by group_id
    def addSinksToGroups(self, group_ids):
        for i, group_id in enumerate(group_ids):
            sink = self.sinks[i]
            self.sinkGroups[group_id]["sinks"].append(sink)
            self.maybeUpdateWireLength(group_id, sink)

    # Checks to see if we have a new longest wire length
    # If we do, we'll update the distance value
    def maybeUpdateWireLength(self, group_id, sink):
        centroid_location = self.sinkGroups[group_id]["centroid_location"]

        # calculate wire length between sink and centroid
        # sqrt((x2 - x1)^2 + (y2 - y1)^2)
        wire_length = self.DistanceToGoal(sink.location, centroid_location)

        # set max distance if applicable
        if (wire_length > self.standardizedWireLength):
            self.standardizedWireLength = wire_length

    # Helper method which finds all factors of a number.
    # This will be used to determine the number of sinks grouped per level in the tree.
    # Algorithm found here: https://stackoverflow.com/a/6800214/5464998
    def findAllFactors(self):
        n = self.num_sinks
        return sorted(set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

    # Draws the plot of tree.
    def makePlot(self):
        self.setAxis()
        self.plotSinkGroups()
        self.showPlot()

    # Sets the axes for our plot.
    def setAxis(self):
        # ([minX, maxX, minY, maxY])
        plt.axis([self.minXCoordinate, 3*self.maxXCoordinate/2, self.minYCoordinate, 3*self.maxYCoordinate/2])

    # Shows our plot
    def showPlot(self):
        plt.show()

    # Plots each of our sink groups
    def plotSinkGroups(self):
        # Get all our plots attributes
        plotAttrs = PlotAttrs()
        centroid_attrs = plotAttrs.centroid_attrs
        sink_attrs = plotAttrs.sink_attrs
        line_colors = plotAttrs.line_colors

        for group_id in self.sinkGroups:
            centroid_location = self.sinkGroups[group_id]["centroid_location"]

            # plot centroid (x, y, attributes)
            plt.plot(
                centroid_location[0],
                centroid_location[1],
                centroid_attrs[group_id]
            )

            # plot sinks (x, y, attributes)
            for sink in self.sinkGroups[group_id]["sinks"]:
                plt.plot(
                    sink.location[0],
                    sink.location[1],
                    sink_attrs[group_id]
                )

                self.drawConnection(sink.location, centroid_location, group_id)

    # draws the line that connects a sink to a centroid
    # takes into account the required (longest) wire length when
    # devising a path
    # wip
    def drawConnection(self, start, goal, group_id):

        # next, we're going to snake the wire from the sink to the centroid,

        desired_distance = self.standardizedWireLength
        exact_distance = self.DistanceToGoal(start, goal)

        # Too close
        if exact_distance != desired_distance:
            self.drawSnakeLine(start, goal, desired_distance, group_id)
        else:
            # Equidistant, so draw direct connection
            self.drawLineBetweenTwoPoints(start, goal, group_id)

    def drawSnakeLine(self, start, goal, goal_wire_length, group):

        bounding_thresholds = {
            "up": self.maxXCoordinate,
            "down": self.minXCoordinate,
            "left": self.minYCoordinate,
            "right": self.maxYCoordinate
        }

        pen = start[:]
        last_pen_position = pen[:]

        direction_traveled = { "up": False, "down": False, "left": False, "right": False }

        turning = False
        go = True
        firstLine = True
        finalLine = False

        current_wire_length = 0

        while go:
            # get relative direction
            direction = self.getSinkRelativeCentroidDirection(start, goal, turning)
            recalculated_direction = direction

            direction_to_travel = self.getNextDirectionToTravel(start, goal, turning)

            movingInPositiveDirection = False if (direction_to_travel == "left" or direction_to_travel == "down") else True
            movingHorizontally = False if (direction_to_travel == "up" or direction_to_travel == "down") else True

            index = 0 if movingHorizontally else 1
            change = 5 if movingInPositiveDirection else -5

            overshoot_counter = 0

            # calculate the length of the wire
            while overshoot_counter < 5:
                # ensure we won't hit the graph bounds
                if movingInPositiveDirection and pen[index] + change > bounding_thresholds[direction_to_travel]:
                    self.directionTraveled[direction] = True
                    break
                elif not movingInPositiveDirection and pen[index] + change < bounding_thresholds[direction_to_travel]:
                    self.directionTraveled[direction] = True
                    break
                # check to see if we've passed the goal on this axis.
                elif recalculated_direction != direction:
                    overshoot_counter += 1
                pen[index] += change
                current_wire_length += abs(change)

                if current_wire_length >= goal_wire_length:
                    finalLine = True
                    break

                recalculated_direction = self.getSinkRelativeCentroidDirection(pen, goal, turning)

            if firstLine:
                print "start to pen"
                self.drawLineBetweenTwoPoints(start, pen, group)
                last_pen_position = pen[:]
                firstLine = False
            elif finalLine:
                print "final line pen"
                self.drawLineBetweenTwoPoints(pen, goal, group)
                go = False
            else:
                print direction
                print self.directionTraveled
                print "pen to pen"
                self.drawLineBetweenTwoPoints(last_pen_position, pen, group)
                last_pen_position = pen[:]

            # toggle our turning variable
            turning = not turning
            self.resetDirectionTravel()


    def drawLineBetweenTwoPoints(self, start, goal, group):
        plotAttrs = PlotAttrs()
        line_colors = plotAttrs.line_colors

        plt.plot(
            [start[0], goal[0]],
            [start[1], goal[1]],
            color=line_colors[group]
        )

    def getSinkRelativeCentroidDirection(self, start, goal, turning=False):
        if turning:
            return "left" if start[0] > goal[0] else "right"
        else:
            return "down" if start[1] > goal[1] else "up"

    def getNextDirectionToTravel(self, start, goal, turning=False):
        if turning:
            possible_dir = "right" if start[0] > goal[0] else "left"
            if self.directionTraveled[possible_dir] == False:
                print "here"
                return possible_dir
            else:
                new_dir = "left" if possible_dir == "right" else "right"

                if self.directionTraveled[new_dir] == False:
                    return new_dir
                else:
                    self.resetDirectionTravel()
                    return getNextDirectionToTravel(start, goal, False)

        else:
            possible_dir =  "up" if start[1] > goal[1] else "down"

            if self.directionTraveled[possible_dir] == False:
                return possible_dir
            else:
                new_dir = "up" if possible_dir == "down" else "down"

                if self.directionTraveled[new_dir] == False:
                    return new_dir
                else:
                    self.resetDirectionTravel()
                    return getNextDirectionToTravel(start, goal, True)

    def DistanceToGoal(self, start, goal):
        return math.sqrt(
            ((goal[0] - start[0]) ** 2) +
            ((goal[1] - start[1]) ** 2)
        )
    def resetDirectionTravel(self):
        print "reset"

    def getDirectionTraveled(self):
        return self.directionTraveled


tree = SymmetricClockTree(50)
tree.generateRandomSinkLocations()
tree.groupSinks()
tree.makePlot()
