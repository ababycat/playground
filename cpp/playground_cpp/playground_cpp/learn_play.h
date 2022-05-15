#pragma once

#include <string>

namespace learn_play{
    class Name{
    public:
        
        std::string name;

        Name(const std::string& name) : name(name){

        }

        std::string GetName(){
            return name;
        }

    };

}