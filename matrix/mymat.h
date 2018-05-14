#pragma once

#include <cstddef>
#include <cmath>
#include <iostream>

#include "constStr.h"
#include "mymat.h"
#include "assertcheck.h"

namespace env{

using std::size_t;
using std::ostream;

// this is a vitural base class
class MyMatBase{
protected:
    size_t rows;
    size_t cols;

public:
    MyMatBase() :rows(0), cols(0){}

    MyMatBase(size_t _rows, size_t _cols)
        :rows(_rows), cols(_cols){}
    
    virtual ~MyMatBase(){}

    friend ostream& operator<<(ostream&, const MyMatBase&);

    size_t getCols() const{ return cols; }
    size_t getRows() const{ return rows; }

    //// property
    virtual bool isSqure() const{ return (rows == cols); }
    //virtual bool isDiagonal() const = 0;
    //virtual bool isSymmetric() const = 0;
    //virtual bool isSparse() const = 0;

    virtual bool set(size_t, size_t, double_t) = 0;
    virtual double_t get(size_t, size_t) const = 0;

    virtual bool Ri_Rj(size_t, size_t) = 0;
    virtual bool Rix_k(size_t, double_t) = 0;
    virtual bool Rki_j(size_t, double_t, size_t) = 0;
    virtual double_t det(void) const = 0;

    // these function will modifiy itself
    virtual MyMatBase& add(const MyMatBase&) = 0;
    virtual MyMatBase& add(const double_t) = 0;
    virtual MyMatBase& sub(const MyMatBase&) = 0;
    virtual MyMatBase& sub(const double_t) = 0;
    virtual MyMatBase& mul(const MyMatBase&) = 0;
    virtual MyMatBase& mul(const double_t) = 0;
};

class DenseMatrix :public MyMatBase{
private:
    double_t** p;
public:
    DenseMatrix();
    
    DenseMatrix(const double_t&);
    DenseMatrix(size_t, size_t);
    DenseMatrix(size_t, size_t, double_t);
    DenseMatrix(const double_t*, size_t, bool);
    DenseMatrix(double_t**, size_t, size_t);

    DenseMatrix(const DenseMatrix&);

    ~DenseMatrix();
    
    friend ostream& operator<<(ostream&, const DenseMatrix&);

    DenseMatrix& operator=(double_t&);
    DenseMatrix& operator=(DenseMatrix&);
    DenseMatrix& operator=(const DenseMatrix&&);

    // these function will modifiy itself
    DenseMatrix& add(const MyMatBase&);
    DenseMatrix& add(const double_t);
    DenseMatrix& sub(const MyMatBase&);
    DenseMatrix& sub(const double_t);
    DenseMatrix& mul(const MyMatBase&);
    DenseMatrix& mul(const double_t);

    bool setCol(size_t inum, double_t* pt, size_t num);
    bool setRow(size_t inum, double_t* pt, size_t num);
    bool set(size_t, size_t, double_t);
    double_t get(size_t, size_t) const;
    
    bool copyTo(DenseMatrix&) const;

    bool Ri_Rj(size_t, size_t);
    bool Rix_k(size_t, double_t);
    bool Rki_j(size_t, double_t, size_t);
    bool Ri_x(size_t i, double_t in);
    double_t det(void) const;

    // these function will NOT modify itself
    const DenseMatrix operator()(size_t, size_t, size_t, size_t);

    const DenseMatrix operator+(const MyMatBase&) const;
    const DenseMatrix operator+(double_t) const;
    const DenseMatrix operator-(const MyMatBase&) const;
    const DenseMatrix operator-(double_t) const;
    const DenseMatrix operator*(const MyMatBase&) const;
    const DenseMatrix operator*(double_t) const;
    const DenseMatrix operator/(const MyMatBase&) const;
    const DenseMatrix operator/(double_t) const;

    const DenseMatrix adj(void) const;
    const DenseMatrix T(void) const;
    const DenseMatrix inv(void) const;
    const DenseMatrix sum(bool) const;
    const DenseMatrix log(void) const;
    const DenseMatrix exp() const;
    const DenseMatrix pow(double_t in) const;

    static DenseMatrix ones(size_t, size_t);
    static DenseMatrix zeros(size_t, size_t);
    static DenseMatrix eyes(size_t);

    DenseMatrix get(size_t, size_t, size_t, size_t) const;



private:
    void clear();
    inline void swap(double_t&, double_t&) const;
    size_t find_num(size_t i) const;
};

}
