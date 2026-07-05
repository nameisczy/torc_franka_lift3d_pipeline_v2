/**
 * @file ompl_test2.cpp
 * @author your name (you@domain.com)
 * @brief 
 * implement the ompl interface with mujoco and hpp-fcl
 * @version 0.1
 * @date 2024-08-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cmath>
#include <cstddef>
#include <mujoco/mujoco.h>
#include "mujoco/mjmodel.h"
#include <mujoco/mjdata.h>

#include <hpp/fcl/collision.h>
#include <hpp/fcl/collision_data.h>
#include <hpp/fcl/BVH/BVH_model.h>

#include <pugixml.hpp>

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
#include <map>
#include <utility>


namespace ob = ompl::base;
namespace og = ompl::geometric;


class RobotValidtyChecker : public ob::StateValidityChecker
{
public:
    /**
     * @brief Construct a new Robot Validty Checker object
     * load the robot mujoco model for forward kinematics, and disabled collisions from robot_srdf_file
     * 
     * @param robot_xml_file 
     * @param robot_srdf_file 
     * @param selected_joint_names 
     * @param si 
     */
    RobotValidtyChecker(const std::string& robot_xml_file, const std::string& robot_srdf_file,
                        const std::map<std::string, double>& default_joint_value_dict,
                        const std::vector<std::string>& selected_joint_names,
                        const ob::SpaceInformationPtr& si) : ob::StateValidityChecker(si)
    {
        /* load the model and data */
        char loadError[1024] = "";
        m = mj_loadXML(robot_xml_file.c_str(), 0, loadError, 1024);
        if (!m)
        {
            mju_error("Could not init model");
        }
        d = mj_makeData(m);

        /* set the default joint values */
        for (auto const& joint_tuple : default_joint_value_dict)
        {
            int joint_id = mj_name2id(m, mjOBJ_JOINT, joint_tuple.first.c_str());
            if (joint_id < 0)
            {
                mju_error("In setting default joints, could not find joint %s", joint_tuple.first.c_str());
            }
            d->qpos[m->jnt_qposadr[joint_id]] = joint_tuple.second;
        }
        mj_forward(m, d);


        /* get the joint information */
        std::vector<int> robot_joint_ids;
        std::vector<std::string> robot_joint_names;
        std::vector<double> robot_joint_lower_limits, robot_joint_upper_limits;
        std::vector<double> robot_joint_values;
        std::vector<int> robot_joint_qposadr;
        for (int i = 0; i < m->njnt; i++)
        {
            robot_joint_names.push_back(m->names + m->name_jntadr[i]);
            robot_joint_lower_limits.push_back(m->jnt_range[i * 2]);
            robot_joint_upper_limits.push_back(m->jnt_range[i * 2 + 1]);
            robot_joint_qposadr.push_back(m->jnt_qposadr[i]);
            robot_joint_values.push_back(d->qpos[m->jnt_qposadr[i]]);
            robot_joint_ids.push_back(i);
        }
        this->robot_joint_names = robot_joint_names;
        this->robot_joint_lower_limits = robot_joint_lower_limits;
        this->robot_joint_upper_limits = robot_joint_upper_limits;
        this->robot_joint_values = robot_joint_values;
        this->robot_joint_qposadr = robot_joint_qposadr;

        /* set the selected joint names and ids */
        std::vector<int> selected_joint_ids;
        std::vector<double> selected_joint_lower_limits;
        std::vector<double> selected_joint_upper_limits;
        std::vector<double> selected_joint_values;
        std::vector<int> selected_joint_qposadr;
        for (size_t i = 0; i < selected_joint_names.size(); i++)
        {
            int joint_id = mj_name2id(m, mjOBJ_JOINT, selected_joint_names[i].c_str());
            if (joint_id < 0)
            {
                mju_error("In setting selected joints, could not find joint %s", selected_joint_names[i].c_str());
            }
            selected_joint_ids.push_back(joint_id);
            selected_joint_lower_limits.push_back(m->jnt_range[joint_id * 2]);
            selected_joint_upper_limits.push_back(m->jnt_range[joint_id * 2 + 1]);
            selected_joint_qposadr.push_back(m->jnt_qposadr[joint_id]);
            selected_joint_values.push_back(d->qpos[m->jnt_qposadr[joint_id]]);
        }
        this->selected_joint_ids = selected_joint_ids;
        this->selected_joint_names = selected_joint_names;
        this->selected_joint_lower_limits = selected_joint_lower_limits;
        this->selected_joint_upper_limits = selected_joint_upper_limits;
        this->selected_joint_values = selected_joint_values;
        this->selected_joint_qposadr = selected_joint_qposadr;

        /* store all the robot link names */
    }

    bool isValid(const ob::State *state) const
    {
        return true;
    }
private:
    mjModel* m = NULL;
    mjData* d = NULL;
    std::vector<int> robot_joint_ids;
    std::vector<std::string> robot_joint_names;
    std::vector<double> robot_joint_values;
    std::vector<int> robot_joint_qposadr;
    std::vector<double> robot_joint_lower_limits;
    std::vector<double> robot_joint_upper_limits;
    std::vector<int> selected_joint_ids;
    std::vector<std::string> selected_joint_names;
    std::vector<double> selected_joint_values;
    std::vector<int> selected_joint_qposadr;
    std::vector<double> selected_joint_lower_limits;
    std::vector<double> selected_joint_upper_limits;


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

    si->setStateValidityChecker(ob::StateValidityCheckerPtr(new RobotValidtyChecker(si)));
    // si->setStateValidityCheckingResolution(0.01);
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