#pragma once

#include <cstddef>
#include <cmath>
#include <iostream>

#include <string>

namespace env{

using std::size_t;
using std::intptr_t;
using std::uint32_t;

using namespace std;

class assert_check{
private:

    inline bool isAequal2B(const size_t A, const size_t B) const{
        return A == B;
    }

public:
    
    static bool isEqual(const size_t A, const size_t B, const char* file, uint32_t line){
        if(A == B){
            return true;
        } else{
            std::cout << "error in file " << file << ", line " << line << '\n'\
                << "Please check the input parameters!\n";
            return false;
        }
    }

    static bool isLessthan(const size_t A, const size_t B, const char* file, uint32_t line){
        if(A < B){
            return true;
        } else{
            std::cout << "error in file " << file << ", line " << line << '\n'\
                << "Please check the input parameters!\n";
            return false;
        }
    }

    static bool isLorE(const size_t A, const size_t B, const char* file, uint32_t line){
        if(A <= B){
            return true;
        } else{
            std::cout << "error in file " << file << ", line " << line << '\n'\
                << "Please check the input parameters!\n";
            return false;
        }
    }

    //template<typename T>
    //inline bool assert_check::isAequal2B(const T A, const T B) const;

    //template<typename T>
    //inline static bool assert_check::isEqual(const T A, const T B, const char* file, uint32_t line);

    //template<typename T>
    //inline static bool assert_check::isEqual2(const T A, const T B, const double i);

};


}

