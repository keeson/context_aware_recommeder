#ifndef CARS_TYPE_H
#define CARS_TYPE_H
#include <vector>
#include <map>

namespace cars
{
    typedef std::vector<double> cars_vector;
    typedef std::vector<cars_vector> cars_matrix;
    typedef std::vector<cars_matrix> cars_tensor;
    typedef std::vector<double> tuple;
    typedef std::vector<tuple> input_type;
    typedef std::map<std::string, unsigned int> id_map;

}

#endif
