#include <iostream>
#include <ceres/ceres.h>

int main() {
    ceres::Problem problem;
    std::vector<ceres::ResidualBlockId> rbs;
    problem.GetResidualBlocks(&rbs);
    return 0;
}