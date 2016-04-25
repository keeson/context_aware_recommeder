#ifndef CARS_TASK_H
#define CARS_TASK_H
#include "cars_type.h"
#include "cars_model.h"
namespace cars
{
    class Task
    {

    protected:
        input_type _input_data;
        Model _model;
        id_map _user_id_map;
        id_map _item_id_map;
        id_map _context_id_map;
        std::vector<std::string> _user_id_backup;
        std::vector<std::string> _item_id_backup;
        std::vector<std::string> _context_id_backup;
    public:
        bool loadInput(char* file_name);
        bool trainModel();
        bool printModel();
    };
        
}
#endif
