#include "cars_model.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <ctime>
#include <omp.h>

using namespace std;
using namespace cars;


Model::Model()
{
}

Model::~Model()
{
}

bool Model::train(input_type & input_data)
{
    cout << _learning_rate <<endl;
    cout << _iter_limit <<endl;

    for (int t = 0; t < 1; t++) {
        for (int i = 0; i < _iter_limit; ++i){
            double error_all = 0.0;
            unsigned count = 0;
            time_t begin_time, end_time;
            time(&begin_time);
            #pragma omp parallel for
            for (int j = 0; j < (input_data.size() / _iter_limit); ++j){
                double error = update(input_data, j*_iter_limit+i);
                #pragma omp atomic
                error_all += error;
                #pragma omp atomic
                ++count;
            }
            cout << "\r\t\t" << i << "\t";

            time(&end_time);
            cout << "||| " << error_all / count << " ||| " << difftime(end_time, begin_time) << "s\n";
        }
        cout << "---------" << t+1 << "-----------" << endl;

    }
}

double Model::update(input_type & input_data, unsigned data_index)
{
    assert(data_index >= 0  && data_index < input_data.size());
    unsigned user_index = unsigned(input_data[data_index][0]);
    if (user_index > 10){
        //return false;
    }
    unsigned item_index = unsigned(input_data[data_index][1]);
    unsigned context_index = unsigned(input_data[data_index][2]);
    double actual_value = input_data[data_index][3];
   
//    return actual_value;
    double predict_value = compute(input_data, data_index);
//    cout << actual_value << ", " << predict_value <<endl;
    double error = predict_value - actual_value;
//    cout<<"user:\n";
//    printVector(_user_model[user_index]);
    updateUserModel(error, user_index, item_index, context_index);
//    printVector(_user_model[user_index]);
//    cout<<"item:\n";
//    printVector(_item_model[item_index]);
    updateItemModel(error, user_index, item_index, context_index);
//    printVector(_item_model[item_index]);
//    cout <<"context:\n";
//    printVector(_context_model[context_index]);
    updateContextModel(error, user_index, item_index, context_index);
//    printVector(_context_model_new[context_index]);
    updateTensorW(error, user_index, item_index, context_index);
    updateTensorZ(error, user_index, item_index, context_index);
    updateUserContextPrefer(error, user_index, item_index, context_index);
    updateItemContextPrefer(error, user_index, item_index, context_index);
    updateSelf(user_index, item_index, context_index);
    return error;
}

bool Model::updateSelf(unsigned user_index, unsigned item_index, unsigned context_index)
{
    replace(_user_model[user_index], _user_model_new[user_index]);
    replace(_item_model[item_index], _item_model_new[item_index]);
    replace(_context_model[context_index], _context_model_new[context_index]);
    replace(_tensor_w, _tensor_w_new);
    replace(_tensor_z, _tensor_z_new);
    replace(_user_context_prefer[user_index], _user_context_prefer_new[user_index]);
    replace(_item_context_prefer[item_index], _item_context_prefer_new[item_index]);

}
bool Model::updateUserModel(double error, unsigned user_index, unsigned item_index, unsigned context_index)
{
    cars_vector temp;
    multiply(_tensor_w, _context_model[context_index], _item_context_prefer[item_index], temp);

//    cout << error << ",";

    #pragma omp parallel for
    for (int i = 0; i < _user_model[user_index].size(); ++i){
        double delta = error * (_item_model[item_index][i] + temp[i]) + _lambda_1 * _user_model[user_index][i];
        //      cout << delta << " ";
        if (delta * 10 > _user_model[user_index][i]){
//            continue;
        }
        _user_model_new[user_index][i] -= _learning_rate * delta;
    }

    return true;
}

bool Model::updateItemModel(double error, unsigned user_index, unsigned item_index, unsigned context_index)
{
    cars_vector temp;
    multiply(_tensor_z, _context_model[context_index], _user_context_prefer[user_index], temp);

    #pragma omp parallel for
    for (int i = 0; i < _item_model[item_index].size(); ++i){
        double delta = error * (_user_model[user_index][i] + temp[i]) + _lambda_1 * _item_model[item_index][i];
        if (delta * 10 > _item_model[item_index][i]){
//            continue;
        }
        _item_model_new[item_index][i] -= _learning_rate * delta;
    }

    return true;
}

bool Model::updateContextModel(double error, unsigned user_index, unsigned item_index, unsigned context_index)
{
    cars_tensor tensor_w_transposed;
    transpose(_tensor_w, tensor_w_transposed);
    cars_tensor tensor_z_transposed;
    transpose(_tensor_z, tensor_z_transposed);
    cars_vector temp_a;
    multiply(tensor_w_transposed, _user_model[user_index], _item_context_prefer[item_index], temp_a);
    cars_vector temp_b;
    multiply(tensor_z_transposed, _item_model[item_index], _user_context_prefer[user_index], temp_b);

    #pragma omp parallel for
    for (int i = 0; i < _context_model[context_index].size(); ++i){
        double delta = 0.0;
        delta += error * (temp_a[i] + temp_b[i]);
//        cout << delta << ", ";
        delta += _lambda_1 * _context_model[context_index][i];
//        cout << delta << "  ";
        if (delta * 10 > _context_model[context_index][i]){
//            continue;
        }
        _context_model_new[context_index][i] -= _learning_rate * delta;
    }
    
    return true;
}

bool Model::updateTensorW(double error, unsigned user_index, unsigned item_index, unsigned context_index)
{
    #pragma omp parallel for
    for (int i = 0; i < _tensor_w.size(); ++i){
        #pragma omp parallel for
        for (int j = 0; j < _tensor_w[i].size(); ++j){
            #pragma omp parallel for
            for (int k = 0; k < _tensor_w[i][j].size(); ++k){
                double delta = error * _item_context_prefer[item_index][j] * _user_model[user_index][i] * _context_model[context_index][k];
                delta += _lambda_2 * _tensor_w[i][j][k];
                if (delta * 10 > _tensor_w[i][j][k]){
//                    continue;
                }
                _tensor_w_new[i][j][k] -= _learning_rate * delta;
//                cout << _tensor_w[i][j][k]  << "\\" << _tensor_w_new[i][j][k] <<endl;
            }
        }
    }

}

bool Model::updateTensorZ(double error, unsigned user_index, unsigned item_index, unsigned context_index)
{
    #pragma omp parallel for
    for (int i = 0; i < _tensor_z.size(); ++i){
        #pragma omp parallel for
        for (int j = 0; j < _tensor_z[i].size(); ++j){
            #pragma omp parallel for
            for (int k = 0; k < _tensor_z[i][j].size(); ++k){
                double delta = error * _user_context_prefer[user_index][j] * _item_model[item_index][i] * _context_model[context_index][k];
                delta += _lambda_2 * _tensor_z[i][j][k];
                if (delta * 10 > _tensor_z[i][j][k]){
//                    continue;
                }
                _tensor_z_new[i][j][k] -= _learning_rate * delta;
//                cout << _tensor_z[i][j][k]  << "\\" << _tensor_z_new[i][j][k] <<endl;
            }
        }
    }

}

bool Model::updateUserContextPrefer(double error, unsigned user_index, unsigned item_index, unsigned context_index)
{
    cars_vector temp;
    multiply(_item_model[item_index], _tensor_z, _context_model[context_index], temp);

    #pragma omp parallel for
    for (int i = 0; i < _user_context_prefer[user_index].size(); ++i){
        double delta = error * temp[i] + _lambda_3 * _user_context_prefer[user_index][i];
        if (delta * 10 > _user_context_prefer[user_index][i]){
//            continue;
        }
        _user_context_prefer_new[user_index][i]  -= _learning_rate * delta;
//        cout << "\\ " << _user_context_prefer[user_index][i]  << " " << _user_context_prefer_new[user_index][i] << " \\";
    }
}

bool Model::updateItemContextPrefer(double error, unsigned user_index, unsigned item_index, unsigned context_index)
{
    cars_vector temp;
    multiply(_user_model[user_index], _tensor_w, _context_model[context_index], temp);

    #pragma omp parallel for
    for (int i = 0; i < _item_context_prefer[item_index].size(); ++i){
        double delta = error * temp[i] + _lambda_3 * _item_context_prefer[item_index][i];
        if (delta * 10 > _item_context_prefer[item_index][i]){
//            continue;
        }
        _item_context_prefer_new[item_index][i] -= _learning_rate * delta;
    }
}


double Model::compute(input_type & input_data, unsigned data_index)
{
    assert(data_index >= 0  && data_index < input_data.size());
    unsigned user_index = unsigned(input_data[data_index][0]);
    unsigned item_index = unsigned(input_data[data_index][1]);
    unsigned context_index = unsigned(input_data[data_index][2]);
    assert(user_index >= 0 && item_index >= 0 && context_index >= 0);
    assert(user_index < _num_u && item_index < _num_v && context_index < _num_c);

    double res = 0.0;
    double tmp = 0.0;
    tmp = multiply(_user_model[user_index], _item_model[item_index]);
    res += tmp;
//    cout << "|" << tmp << " | ";
    tmp = multiply(_item_context_prefer[item_index],
                    _user_model[user_index],
                    _tensor_w,
                    _context_model[context_index]);
    res += tmp;
//    cout << tmp << " | ";
    tmp = multiply(_user_context_prefer[user_index],
                    _item_model[item_index],
                    _tensor_z,
                    _context_model[context_index]);
    res += tmp;
//    cout << tmp << "|\n";

    return res;

}

bool Model::set_parameter(unsigned num_u, unsigned num_v, unsigned num_c,
                          double lambda_1,
                          double lambda_2,
                          double lambda_3,
                          unsigned int iter_limit,
                          double learning_rate)
                          
{
    _lambda_1 = lambda_1;
    _lambda_2 = lambda_2;
    _lambda_3 = lambda_3;
    _iter_limit = iter_limit;
    _learning_rate = learning_rate;
    _num_u = num_u;
    _num_v = num_v;
    _num_c = num_c;
    return true;
}

bool Model::initialize()
{
    
    _dimension_uv = DIMENSION_UV_DEFAULT;
    _dimension_p = DIMENSION_P_DEFAULT;
    _dimension_q = DIMENSION_Q_DEFAULT;
    _dimension_c = DIMENSION_C_DEFAULT;
    _iter_limit = ITER_LIMIT_DEFAULT;

    assert(_num_u > 0 && _num_v > 0 && _num_c > 0);

    _user_model_new.resize(_num_u);
    _user_model.resize(_num_u);
    #pragma omp parallel for
    for (int i = 0; i < _user_model.size(); ++i){
        _user_model_new[i].resize(_dimension_uv);
        _user_model[i].resize(_dimension_uv);
        initializeVector(_user_model[i]);
        replace(_user_model_new[i], _user_model[i]);
    }
    _item_model.resize(_num_v);
    _item_model_new.resize(_num_v);

    #pragma omp parallel for
    for (int i = 0; i < _item_model.size(); ++i){
        _item_model[i].resize(_dimension_uv);
        _item_model_new[i].resize(_dimension_uv);
        initializeVector(_item_model[i]);
        replace(_item_model_new[i], _item_model[i]);
    }
    _context_model.resize(_num_c);
    _context_model_new.resize(_num_c);
    #pragma omp parallel for
    for (int i = 0; i < _context_model.size(); ++i){
        _context_model[i].resize(_dimension_c);
        _context_model_new[i].resize(_dimension_c);
        initializeVector(_context_model[i]);
        replace(_context_model_new[i], _context_model[i]);
    }
    _user_context_prefer.resize(_num_u);
    _user_context_prefer_new.resize(_num_u);
    #pragma omp parallel for
    for (int i = 0; i < _user_context_prefer.size(); ++i){
        _user_context_prefer[i].resize(_dimension_q);
        _user_context_prefer_new[i].resize(_dimension_q);
        initializeVector(_user_context_prefer[i]);
        replace(_user_context_prefer_new[i], _user_context_prefer[i]);
    }
    _item_context_prefer.resize(_num_v);
    _item_context_prefer_new.resize(_num_v);
    #pragma omp parallel for
    for (int i = 0; i < _item_context_prefer.size(); ++i){
        _item_context_prefer[i].resize(_dimension_p);
        _item_context_prefer_new[i].resize(_dimension_p);
        initializeVector(_item_context_prefer[i]);
        replace(_item_context_prefer_new[i], _item_context_prefer[i]);
    }

    _tensor_w.resize(_dimension_uv);
    _tensor_w_new.resize(_dimension_uv);
    #pragma omp parallel for
    for (int i = 0; i < _tensor_w.size(); ++i){
        _tensor_w[i].resize(_dimension_p);
        _tensor_w_new[i].resize(_dimension_p);
        #pragma omp parallel for
        for (int j = 0; j < _tensor_w[i].size(); ++j){
            _tensor_w[i][j].resize(_dimension_c);
            _tensor_w_new[i][j].resize(_dimension_c);
            initializeVector(_tensor_w[i][j]);
        }
        
    }
    _tensor_z.resize(_dimension_uv);
    _tensor_z_new.resize(_dimension_uv);
    #pragma omp parallel for
    for (int i = 0; i < _tensor_z.size(); ++i){
        _tensor_z[i].resize(_dimension_q);
        _tensor_z_new[i].resize(_dimension_q);
        #pragma omp parallel for
        for (int j = 0; j < _tensor_z[i].size(); ++j){
            _tensor_z[i][j].resize(_dimension_c);
            _tensor_z_new[i][j].resize(_dimension_c);
            initializeVector(_tensor_z[i][j]);
        }
    }

    replace(_tensor_w_new, _tensor_w);
    replace(_tensor_z_new, _tensor_z);
    
}
