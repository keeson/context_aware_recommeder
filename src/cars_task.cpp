#include "cars_task.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace cars;

bool Task::loadInput(char* file_name)
{
    _input_data.clear();
    ifstream fin;
    fin.open(file_name);
    unsigned record_index = 0;
    while (!fin.eof()){
        string user_id, item_id, context_id;
        fin >> user_id >> item_id >> context_id;
        unsigned user_index, item_index, context_index;
        if (_user_id_map.find(user_id) != _user_id_map.end()){
            user_index = _user_id_map[user_id];
        }else{
            user_index = _user_id_backup.size();
            _user_id_backup.push_back(user_id);
            _user_id_map[user_id] = user_index;
        }

        if (_item_id_map.find(item_id) != _item_id_map.end()){
            item_index = _item_id_map[item_id];
        }else{
            item_index = _item_id_backup.size();
            _item_id_backup.push_back(item_id);
            _item_id_map[item_id] = item_index;
        }

        if (_context_id_map.find(context_id) != _context_id_map.end()){
            context_index = _context_id_map[context_id];
        }else{
            context_index = _context_id_backup.size();
            _context_id_backup.push_back(context_id);
            _context_id_map[context_id] = context_index;
        }

        double rate;
        fin >> rate;

        tuple temp;
        temp.resize(4);
        temp[0] = user_index;
        temp[1] = item_index;
        temp[2] = context_index;
        temp[3] = rate * 10;
        _input_data.push_back(temp);
        cout << "\r " << record_index++;
    }
    cout << "read complete." <<endl;
    return true;
}

bool Task::trainModel()
{
    _model.set_parameter(_user_id_backup.size(),
                         _item_id_backup.size(),
                         _context_id_backup.size());
    _model.initialize();

    for (int i = 0; i < 500; ++i){
        _model.train(_input_data);
        printModel();
    }
    return true;
}

bool Task::printModel()
{
    cout << _model._dimension_uv <<endl;
    cout << _model._dimension_c <<endl;
    cout << _model._dimension_p <<endl;
    cout << _model._dimension_q <<endl;

    ofstream fout;
    fout.open("user_model", std::ofstream::out);
    for (int i = 0; i < _model._user_model.size(); ++i){
        fout << _user_id_backup[i] << ":";
        printVector(_model._user_model[i], fout);
        fout << "\n";
    }
    fout.close();
    fout.open("item_model", std::ofstream::out);
    for (int i = 0; i < _model._item_model.size(); ++i){
        fout << _item_id_backup[i] << ":";
        printVector(_model._item_model[i], fout);
        fout << "\n";
    }
    fout.close();

    fout.open("context_model", std::ofstream::out);
    for (int i = 0; i < _model._context_model.size(); ++i){
        fout << _context_id_backup[i] << ":";
        printVector(_model._context_model[i], fout);
        fout << "\n";
    }
    fout.close();
    fout.open("user_context_prefer", std::ofstream::out);
    for (int i = 0; i < _model._user_model.size(); ++i){
        fout << _user_id_backup[i] << ":";
        printVector(_model._user_context_prefer[i], fout);
        fout << "\n";
    }
    fout.close();

    fout.open("item_context_prefer", std::ofstream::out);
    for (int i = 0; i < _model._item_model.size(); ++i){
        fout << _item_id_backup[i] << ":";
        printVector(_model._item_context_prefer[i], fout);
        fout << "\n";
    }
    fout.close();

    fout.open("tensor_w", std::ofstream::out);
    printTensor(_model._tensor_w, fout);
    fout.close();

    fout.open("tensor_z", std::ofstream::out);
    printTensor(_model._tensor_z, fout);
    fout.close();
    
}
