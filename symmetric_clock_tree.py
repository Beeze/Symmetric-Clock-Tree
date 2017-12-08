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

    # Helper method to calculate the distance between two points.
    def DistanceToGoal(self, start, goal):
        return math.sqrt(
            ((goal[0] - start[0]) ** 2) +
            ((goal[1] - start[1]) ** 2)
        )

    # reset's our directionTraveled Dictionary
    def resetDirectionTravel(self):
        self.directionTraveled = { "up": False, "down": False, "left": False, "right": False }

    # getter method for get
    def getDirectionTraveled(self):
        return self.directionTraveled

    # Helper method for generating random sink locations.
    def generateRandomSinkLocations(self):
        radius = 1000
        rangeX = (self.maxXCoordinate/4, (3*self.maxXCoordinate/4))
        rangeY = (self.maxYCoordinate/4, (3*self.maxYCoordinate/4))
        qty = self.num_sinks  # or however many points you want

        # Generate a set of all points within 200 of the origin, to be used as offsets later
        # There's probably a more efficient way to do this.
        deltas = set()
        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                if x*x + y*y <= radius*radius and x*x <= (self.maxXCoordinate*2) and y*y <= (self.maxYCoordinate):
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
        plt.axis([self.minXCoordinate, (self.maxXCoordinate + self.maxXCoordinate/5), self.minYCoordinate, (self.maxYCoordinate + self.maxXCoordinate/5) ])

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

        # turning tracks which axis we're traveling on.
        turning = False

        continue_drawing_line = True

        # booleans that track which point we're going to or coming from.
        connect_to_start = True
        connect_to_goal = False

        # Initialize our wire length
        current_wire_length = 0

        while continue_drawing_line:
            # get relative direction
            direction = self.getCurrentLocationRelativeToCentroid(start, goal, turning)

            # initialize recalculated_direction
            recalculated_direction = direction

            # determine the best direction to begin drawing
            direction_to_travel = self.getNextDirectionToTravel(start, goal, turning)

            # movingInPositiveDirection tracks which axis we're moving in
            movingInPositiveDirection = False if (direction_to_travel == "left" or direction_to_travel == "down") else True
            # movingHorizontally tracks which direction we're moving in
            movingHorizontally = False if (direction_to_travel == "up" or direction_to_travel == "down") else True

            # index tracks which pen position we're changing (x or y)
            index = 0 if movingHorizontally else 1

            # change sets the amount we increment / decrement each time.
            change = 5 if movingInPositiveDirection else -5

            # overshoot_counter tracks how many times we've crossed from the pos -> neg axis, relative to the goal location.
            overshoot_counter = 0

            # we allow ourselves to cross the axis 5 times
            while overshoot_counter < 5:
                # check if we've hit the graph bounds.
                # if we have, we'll trip a boolean so we don't keep trying the same direction
                if movingInPositiveDirection and pen[index] + change > bounding_thresholds[direction_to_travel]:
                    self.directionTraveled[direction] = True
                    break
                elif not movingInPositiveDirection and pen[index] + change < bounding_thresholds[direction_to_travel]:
                    self.directionTraveled[direction] = True
                    break

                # check to see if we've passed the goal on this axis.
                elif recalculated_direction != direction:
                    overshoot_counter += 1

                # update the pen location and current_wire_length
                pen[index] += change
                current_wire_length += abs(change)

                # check if we've reached our goal wire length
                if current_wire_length >= goal_wire_length:
                    connect_to_goal = True
                    break

                # check our pen's location relative to the centroid
                recalculated_direction = self.getCurrentLocationRelativeToCentroid(pen, goal, turning)

            # Check what type of point we're connecting
            if connect_to_start:
                self.drawLineBetweenTwoPoints(start, pen, group)
                last_pen_position = pen[:]
                connect_to_start = False

            elif connect_to_goal:
                self.drawLineBetweenTwoPoints(pen, goal, group)
                continue_drawing_line = False

            else:
                self.drawLineBetweenTwoPoints(last_pen_position, pen, group)
                last_pen_position = pen[:]

            # toggle our turning variable
            turning = not turning
            self.resetDirectionTravel()

    # Helper method to draw a line between two points
    def drawLineBetweenTwoPoints(self, start, goal, group):
        plotAttrs = PlotAttrs()
        line_colors = plotAttrs.line_colors

        plt.plot(
            [start[0], goal[0]],
            [start[1], goal[1]],
            color=line_colors[group]
        )

    # Helper method which calculates current location relative to centroid
    def getCurrentLocationRelativeToCentroid(self, start, goal, turning=False):
        if turning:
            return "left" if start[0] > goal[0] else "right"
        else:
            return "down" if start[1] > goal[1] else "up"

    # Helper method to determine which direction we'll travel in
    def getNextDirectionToTravel(self, start, goal, turning=False):

        # Algorithm checks to see which direction we have the most space to travel in.
        # Once one is selected, we do a lookup to see if we've already tried to go in
        # that direction during this iteration.
        # If we have, we try the opposite direction.
        # If we've tried both, we'll call the function and move in a different
        # orientation.
        if turning:
            possible_dir = "right" if start[0] > goal[0] else "left"
            if self.directionTraveled[possible_dir] == False:
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


tree = SymmetricClockTree(50)
tree.generateRandomSinkLocations()
tree.groupSinks()
tree.makePlot()
