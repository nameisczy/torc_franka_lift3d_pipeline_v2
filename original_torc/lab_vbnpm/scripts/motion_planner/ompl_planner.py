import numpy as np
from ompl import base as ob
from ompl import geometric as og


class SpherePlanner():

    def isStateValid(self, state):

        # collision check
        return self.collision_free(state)

    def __init__(self, radius, points, ws_size):
        # create an R^3 state space
        space = ob.RealVectorStateSpace(3)

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(-ws_size)
        bounds.setHigh(ws_size)
        space.setBounds(bounds)

        # create a simple setup object
        si = ob.SpaceInformation(space)
        si.setStateValidityCheckingResolution(0.001)
        ss = og.SimpleSetup(si)
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        print(ss.getSpaceInformation().settings())

        # expose relevant variables
        self.ss = ss
        self.start = ob.State(space)
        self.goal = ob.State(space)
        self.set_points(points)

    def set_points(self, points):
        self.kd_tree = ob.KDTree(3)

    def plan(self, start, goal, time=1.0, TypePlanner=og.RRTConnect):
        for i in range(self.dof):
            self.start[i] = start[i]
            self.goal[i] = goal[i]
        self.ss.setStartAndGoalStates(self.start, self.goal)
        self.ss.setPlanner(TypePlanner(self.ss.getSpaceInformation()))

        # this will automatically choose a default planner with default parameters
        solved = self.ss.solve(time)
        if solved:
            # try to shorten the path
            self.ss.simplifySolution()
            # print the simplified path
            states = self.ss.getSolutionPath().getStates()
            return [[state[i] for i in range(self.dof)] for state in states]
        return []


if __name__ == "__main__":
    planner = Planner(6, [-1.58] * 6, [1.58] * 6)
    plan = planner.plan(
        np.random.uniform(-1.57, 1.57, 6), np.random.uniform(-1.57, 1.57, 6)
    )
    print(plan)
