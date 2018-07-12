#pragma once

#include <cstddef>
#include <cmath>
#include <iostream>
#include <string>
namespace env{

class Mat{
private:
    std::size_t rows;
    std::size_t cols;
    std::double_t **p;
public:
    Mat();
    Mat(std::size_t, std::size_t);
    Mat(std::size_t _rows, std::size_t _cols, std::double_t value);
    ~Mat();

    bool isMatInit() const;

    const Mat& operator=(const Mat&) const;

    //void setRows(std::size_t);
    //void setCols(std::size_t);

    void clear();

    std::size_t getRows() const;
    std::size_t getCols() const;

    friend std::ostream& operator<<(std::ostream & output, Mat& _A);

    void cout() const;
};


}
