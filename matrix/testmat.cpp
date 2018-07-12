#include "stdafx.h"
#include "testmat.h"
namespace env
{

Mat::Mat()
    :rows(0), cols(0)
{
    p = nullptr;
}

Mat::Mat(std::size_t _rows, std::size_t _cols)
    : rows(_rows), cols(_cols)
{
    p = new std::double_t*[_rows];
    for(std::size_t i = 0; i < _rows; ++i)
    {
        p[i] = new std::double_t[_cols]{0};
    }
}

Mat::Mat(std::size_t _rows, std::size_t _cols, std::double_t value)
    : rows(_rows), cols(_cols)
{
    p = new std::double_t*[_rows];
    for(std::size_t i = 0; i < _rows; ++i)
    {
        p[i] = new std::double_t[_cols];
        for(std::size_t j = 0; j < _cols; ++j)
        {
            p[i][j] = value;
        }
    }
}

Mat::~Mat()
{
    if(p != nullptr)
    {
        for(std::size_t i = 0; i < rows; ++i)
        {
            delete[] p[i];
        }
        delete[] p;
    }
}

bool Mat::isMatInit() const
{
    return (p != nullptr) && (*p != nullptr);
}

const Mat & Mat::operator=(const Mat &in) const
{
    return in;
}

//void Mat::setRows(std::size_t _rows)
//{
//    // clear ptr
//    rows = _rows;
//}
//
//void Mat::setCols(std::size_t _cols)
//{
//    cols = _cols;
//}
//clear the Mat
void Mat::clear()
{
}
//get the rows
std::size_t Mat::getRows() const
{
    return rows;
}
//get the cols
std::size_t Mat::getCols()  const
{
    return cols;
}
// this can write to file or display in cmd
std::ostream & operator<<(std::ostream & output, Mat& _A)
{
    for(std::size_t i = 0; i < _A.getRows(); ++i)
    {
        output << "     ";
        for(std::size_t j = 0; j < _A.getCols(); ++j)
        {
            output << _A.p[i][j] << " , ";
        }
        output << "\n";
    }
    output << "\n";
    return output;
}
// this function only can display in cmd
void Mat::cout() const
{
    for(std::size_t i = 0; i < rows; ++i)
    {
        std::cout << "    ";
        for(std::size_t j = 0; j < cols; ++j)
        {
            std::cout << p[i][j] << " , ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

}// env
