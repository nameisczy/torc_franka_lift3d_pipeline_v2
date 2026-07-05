/**
 * @file ompl_test.cpp
 * @author your name (you@domain.com)
 * @brief 
 * implement the basic functionalites of ompl interface, without other packages (Mujoco and hpp-fcl).
 * @version 0.1
 * @date 2024-08-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/goals/GoalStates.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/Planner.h>
#include <ompl/base/PlannerStatus.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

#include <iostream>


namespace ob = ompl::base;
namespace og = ompl::geometric;

class myStateValidityChecker : public ob::StateValidityChecker
{
public:
    myStateValidityChecker(const ob::SpaceInformationPtr &si) : ob::StateValidityChecker(si) {}
    bool isValid(const ob::State *state) const
    {
        return true;
    }
};  


int main(int argc, char** argv)
{
    ob::StateSpacePtr space(new ob::RealVectorStateSpace(8));
    ob::RealVectorBounds bounds(8);
    bounds.setLow(0, -3.14); bounds.setHigh(0, 3.14);
    bounds.setLow(1, -3.14); bounds.setHigh(1, 3.14);
    bounds.setLow(2, -3.14); bounds.setHigh(2, 3.14);
    bounds.setLow(3, -3.14); bounds.setHigh(3, 3.14);
    bounds.setLow(4, -3.14); bounds.setHigh(4, 3.14);
    bounds.setLow(5, -3.14); bounds.setHigh(5, 3.14);
    bounds.setLow(6, -3.14); bounds.setHigh(6, 3.14);
    bounds.setLow(7, -3.14); bounds.setHigh(7, 3.14);
    space->as<ob::RealVectorStateSpace>()->setBounds(bounds);
    ob::SpaceInformationPtr si(new ob::SpaceInformation(space));

    si->setStateValidityChecker(ob::StateValidityCheckerPtr(new myStateValidityChecker(si)));
    // si->setStateValidityCheckingResolution(0.01);
    // 1.0 / space->getMaximumExtent()
    si->setup();

    ob::ProblemDefinitionPtr pdef(new ob::ProblemDefinition(si));
    ob::ScopedState<ob::RealVectorStateSpace> start(space);
    start.random();
    ob::ScopedState<ob::RealVectorStateSpace> goal(space);
    goal.random();
    pdef->addStartState(start);
    double threshold = 0.01;
    pdef->setGoalState(goal, threshold);
    // pdef->setOptimizationObjective(ob::OptimizationObjectivePtr(new ob::PathLengthOptimizationObjective(si)));

    /* intialize planner */
    ob::PlannerPtr planner(new og::RRTConnect(si));
    planner->setProblemDefinition(pdef);
    planner->setup();
    pdef->print(std::cout);

    /* solve the problem */
    ob::PlannerStatus solved = planner->solve(1.0);
    if (solved)
    {
        std::cout << "status: " << solved.asString() << std::endl;
        pdef->getSolutionPath()->print(std::cout);
    }
    else
    {
        std::cout << "status: " << solved.asString() << std::endl;
        std::cout << "No solution found" << std::endl;
    }

}