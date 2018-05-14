#include "stdafx.h"
#include "mystring.h"

namespace env{

std::size_t strpro::find_num(std::string& s, bool reserve_flag = 0){
    if(reserve_flag == 0){
        for(std::size_t i = 0; i < s.length(); ++i){
            if((s.at(i) >= '0' && s.at(i) <= '9') || (s.at(i) == '.') || (s.at(i) == '-')){
                return i;
            }
        }
    } else if(reserve_flag == 1){
        for(std::intptr_t i = s.length() - 1; i >= 0; --i){
            if((s.at(i) >= '0' && s.at(i) <= '9') || (s.at(i) == '.') || (s.at(i) == '-')){
                return i;
            }
        }
    }
    return s.length();
}

void strpro::lstrip(std::string& str){
    str = str.substr(find_num(str));
}

std::size_t strpro::find_t_s(std::string& s, std::size_t pos = 0){
    for(std::size_t i = 0 + pos; i < s.length(); ++i){
        if(s.at(i) == '\t' || s.at(i) == ' '){
            return i;
        }
    }
    return s.length();
}

int strpro::split(std::string& s, std::vector<std::string>& svec){
    int i = 0;
    while(s.length() != 0 && find_num(s) != s.length()){
        lstrip(s);
        svec.push_back(s.substr(0, find_t_s(s)));
        ++i;
        s = s.substr(find_t_s(s));
    }
    return i;
}

int strpro::split(std::string& s, std::vector<std::double_t>& vec){
    //lstrip(s);
    //vec.push_back(std::stod(s.substr(0, find_t_s(s))));
    //s = s.substr(find_t_s(s));

    //lstrip(s);
    //vec.push_back(std::stod(s.substr(0, find_t_s(s))));
    //s = s.substr(find_t_s(s));
    int i = 0;
    while(s.length() != 0 && find_num(s) != s.length()){
        lstrip(s);
        vec.push_back(std::stod(s.substr(0, find_t_s(s))));
        i++;
        s = s.substr(find_t_s(s));
    }
    return i;
}

void strpro::display(std::vector<std::double_t>& vec){
    std::cout << std::fixed << std::setprecision(8);
    for(std::vector<std::double_t>::iterator p = vec.begin();
    p != vec.end(); ++p){
        std::cout << *p << ",   ";
    }
    std::cout << std::endl;
}

}
