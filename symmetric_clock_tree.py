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
        self.maxXCoordinate = 2500
        self.minYCoordinate = 0
        self.maxYCoordinate = 2500
        self.standardizedWireLength = 0

    # Helper method for generating random sink locations.
    def generateRandomSinkLocations(self):
        radius = 200
        rangeX = (self.minXCoordinate, self.maxXCoordinate)
        rangeY = (self.minYCoordinate, self.maxYCoordinate)
        qty = self.num_sinks  # or however many points you want

        # Generate a set of all points within 200 of the origin, to be used as offsets later
        # There's probably a more efficient way to do this.
        deltas = set()
        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                if x*x + y*y <= radius*radius:
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
        wire_length = math.sqrt(
            ((centroid_location[0] - sink.location[0]) ** 2) +
            (centroid_location[1] - sink.location[1]) ** 2)

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
        plt.axis([self.minXCoordinate, self.maxXCoordinate, self.minYCoordinate, self.maxYCoordinate])

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

                self.drawConnection(centroid_location, sink.location)

    # draws the line that connects a sink to a centroid
    # takes into account the required (longest) wire length when
    # devising a path
    def drawConnection(self, centroid_location, sink_location):

        # next, we're going to snake the wire from the sink to the centroid,
        # we decide which horizontal and vertical direction to go
        # based on whichever side we have more clearance on.
        sink_relative_centroid_direction = getSinkRelativeCentroidDirection(centroid_location, sink_location)
        wire_length = self.standardizedWireLength
        pen_location = sink.location
        drawSnakeLine(pen_location, wire_length, sink_relative_centroid_direction)
        # connect sink to their centroid. ([x_start, x_end], [y_start, y_end], attributes)
        # plt.plot(
        #     [sink.location[0], centroid_location[0]],
        #     [sink.location[1], centroid_location[1]],
        #     color=line_colors[group_id]
        # )

    def getSinkRelativeCentroidDirection(self, c_l, s_l):
        if c_l[0] > s_l[0] and c_l[1] > s_l[1]
            return 'NE'
        elif c_l[0] > s_l[0] and c_l[1] < s_l[1]
            return 'NW'
        elif c_l[0] < s_l[0] and c_l[1] > s_l[1]
            return 'SE'
        else
            return 'SW'

    def drawSnakeLine(self, starting_location, distance_to_travel, direction):


    # Shows our plot
    def showPlot(self):
        plt.show()


tree = SymmetricClockTree(49)
tree.generateRandomSinkLocations()
tree.groupSinks()
tree.makePlot()
