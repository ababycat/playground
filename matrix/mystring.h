#pragma once

#include <cmath>
#include <iostream>
#include <cstddef>
#include <vector>
#include <string>
#include <iomanip>

namespace env{

class strpro{
public:
static int split(std::string& s, std::vector<std::string>& svec);
static void display(std::vector<std::double_t>& vec);
static int split(std::string& s, std::vector<std::double_t>& vec);

private:
static std::size_t find_t_s(std::string& s, std::size_t);
static std::size_t find_num(std::string& s, bool);
static void lstrip(std::string& str);

};

}
