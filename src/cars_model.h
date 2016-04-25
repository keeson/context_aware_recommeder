#ifndef CARS_MODEL_H
#define CARS_MODEL_H

#include "cars_type.h"
#include "cars_calculate.h"

namespace cars
{

    const unsigned DIMENSION_UV_DEFAULT = 10;
    const unsigned DIMENSION_P_DEFAULT = 4;
    const unsigned DIMENSION_Q_DEFAULT = 8;
    const unsigned DIMENSION_C_DEFAULT = 4;
    const unsigned ITER_LIMIT_DEFAULT = 1;
    const double LAMBDA_DEFAULT = 0.001;
    const double LEARNING_RATE_DEFAULT = 0.01;
    class Task;

    class Model
    {
    public:
        friend  class Task;
    public:
        Model();
        ~Model();
        bool set_parameter(unsigned num_u, unsigned num_v, unsigned num_c,
                           double lambda_1 = LAMBDA_DEFAULT,
                           double lambda_2 = LAMBDA_DEFAULT,
                           double lambda_3 = LAMBDA_DEFAULT,
                           unsigned int iter_limit = ITER_LIMIT_DEFAULT,
                           double learning_rate = LEARNING_RATE_DEFAULT);
        bool train(input_type & input_data);
        bool initialize();


    protected:
        double update(input_type & input_data, unsigned int data_index);
        double compute(input_type & input_data, unsigned int data_index);
        bool updateSelf(unsigned user_index, unsigned item_index, unsigned context_index);
        bool updateUserModel(double error, unsigned user_index, unsigned item_index, unsigned context_index);
        bool updateItemModel(double error, unsigned user_index, unsigned item_index, unsigned context_index);
        bool updateContextModel(double error, unsigned user_index, unsigned item_index, unsigned context_index);
        bool updateTensorW(double error, unsigned user_index, unsigned item_index, unsigned context_index);
        bool updateTensorZ(double error, unsigned user_index, unsigned item_index, unsigned context_index);
        bool updateUserContextPrefer(double error, unsigned user_index, unsigned item_index, unsigned context_index);
        bool updateItemContextPrefer(double error, unsigned user_index, unsigned item_index, unsigned context_index);    
            
    protected:
        std::vector<cars_vector> _user_model;
        std::vector<cars_vector> _item_model;
        std::vector<cars_vector> _context_model;
        std::vector<cars_vector> _user_context_prefer;
        std::vector<cars_vector> _item_context_prefer;
        cars_tensor _tensor_w;
        cars_tensor _tensor_z;
        double _lambda_1, _lambda_2, _lambda_3;
        double _learning_rate;
        unsigned int _dimension_uv, _dimension_p, _dimension_q, _dimension_c;
        unsigned int _iter_limit;
        unsigned int _num_u, _num_v, _num_c;

        std::vector<cars_vector> _user_model_new;
        std::vector<cars_vector> _item_model_new;
        std::vector<cars_vector> _context_model_new;
        std::vector<cars_vector> _user_context_prefer_new;
        std::vector<cars_vector> _item_context_prefer_new;
        cars_tensor _tensor_w_new;
        cars_tensor _tensor_z_new;
    };

}

#endif
