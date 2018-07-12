#include "stdafx.h"
#include "mymat.h"

namespace env{
//// Mat Base
//void MyMatBase::setRows(size_t _rows){
//    rows = _rows;
//}
//
//void MyMatBase::setCols(size_t _cols){
//    cols = _cols;
//}

DenseMatrix::DenseMatrix()
    :MyMatBase(), p(nullptr){}

DenseMatrix::DenseMatrix(const double_t& in)
    : MyMatBase(1, 1), p(nullptr){
    p = new double_t*[rows];
    p[0] = new double_t[cols]{in};
}

DenseMatrix::DenseMatrix(size_t _rows, size_t _cols)
    : MyMatBase(_rows, _cols), p(nullptr){
    p = new double_t*[_rows];
    for(size_t i = 0; i < _rows; ++i){
        p[i] = new double_t[_cols]{0};
    }
}

DenseMatrix::DenseMatrix(size_t _rows, size_t _cols, double_t value)
    :MyMatBase(_rows, _cols), p(nullptr){
    p = new double_t*[_rows];
    for(size_t i = 0; i < _rows; ++i){
        p[i] = new double_t[_cols];
        for(size_t j = 0; j < _cols; ++j){
            p[i][j] = value;
        }
    }
}
// if
DenseMatrix::DenseMatrix(const double_t *in, size_t length, bool vec_type)
    :MyMatBase(vec_type ? length : 1, vec_type ? 1 : length), p(nullptr){
    if(vec_type){
        p = new double_t*[rows];
        for(intptr_t i = rows - 1; i >= 0; --i){
            p[i] = new double_t[1];
            p[i][0] = in[i];
        }
    } else{
        p = new double_t*[1];
        p[0] = new double_t[cols];
        for(intptr_t i = cols - 1; i >= 0; --i){
            p[0][i] = in[i];
        }
    }
}

//DenseMatrix::DenseMatrix(const double_t **in, size_t _rows, size_t _cols)
//    :MyMatBase(_rows, _cols),p(nullptr){
//    p = new double_t*[_rows];
//    for(size_t i = 0; i < rows; ++i){
//        p[i] = new double_t[cols];
//        for(size_t j = 0; j < cols; ++j){
//            p[i][j] = in[i][j];
//        }
//    }
//}

DenseMatrix::DenseMatrix(const DenseMatrix& in)
    :MyMatBase(in.rows, in.cols), p(nullptr){
    p = new double_t*[rows];
    for(size_t i = 0; i < rows; ++i){
        p[i] = new double_t[cols];
        //p[i] = in.p[i];
        for(size_t j = 0; j < cols; ++j){
            p[i][j] = in.p[i][j];
        }
    }
}

DenseMatrix::~DenseMatrix(){
    if(p != nullptr){
        clear();
    }
}

void DenseMatrix::clear(){
    for(size_t i = 0; i < rows; ++i){
        delete[] p[i];
        p[i] = nullptr;
    }
    delete[] p;
    p = nullptr;
}

DenseMatrix& DenseMatrix::add(const MyMatBase& in){
    assert_check::isEqual(rows, in.getRows(), __FILE__, __LINE__);
    assert_check::isEqual(cols, in.getCols(), __FILE__, __LINE__);
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            p[i][j] += in.get(i, j);
        }
    }
    return *this;
}

DenseMatrix& DenseMatrix::add(const double_t in){
    if(in == 0){
        return *this;
    }
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            p[i][j] += in;
        }
    }
    return *this;
}

DenseMatrix& DenseMatrix::sub(const MyMatBase& in){
    assert_check::isEqual(rows, in.getRows(), __FILE__, __LINE__);
    assert_check::isEqual(cols, in.getRows(), __FILE__, __LINE__);
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            p[i][j] -= in.get(i, j);
        }
    }
    return *this;
}

DenseMatrix& DenseMatrix::sub(double_t in){
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            p[i][j] -= in;
        }
    }
    return *this;
}

DenseMatrix& DenseMatrix::mul(double_t in){
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            p[i][j] *= in;
        }
    }
    return *this;
}

DenseMatrix& DenseMatrix::mul(const MyMatBase& in){
    assert_check::isEqual(cols, in.getRows(), __FILE__, __LINE__);
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            p[i][j] *= in.get(i, j);
        }
    }
    //    for(intptr_t i = out.rows - 1; i >= 0; --i){
    //        for(intptr_t j = out.cols - 1; j >= 0; --j){
    //            for(intptr_t k = out.cols - 1; k >= 0; --k){
    //                out.p[i][j] += p[i][k] * in.p[k][j];
    //            }
    //        }
    //    }
    return *this;
}

//
DenseMatrix& DenseMatrix::operator=(double_t& in){
    if(p != nullptr){
        clear();
    }

    rows = 1;
    cols = 1;

    p = new double_t*[rows];
    p[0] = new double_t[cols]{in};

    return *this;
}
// remember this function only copy the pointer
// but if you have moved the original data will released
DenseMatrix& DenseMatrix::operator=(DenseMatrix& in){
    // release the memory
    if(this != &in){
        if(p != nullptr){
            clear();
        }
    }
    rows = in.rows;
    cols = in.cols;
    // copy the pointer
    p = in.p;
    // release the original memory
    in.p = nullptr;
    return *this;
}
// remember this function only copy the pointer
DenseMatrix& DenseMatrix::operator=(const DenseMatrix&& in){
    // release the memory
    if(this != &in){
        if(p != nullptr){
            clear();
        }
    }
    rows = in.rows;
    cols = in.cols;

    p = new double_t*[rows];
    for(size_t i = 0; i < rows; ++i){
        p[i] = new double_t[cols];
        for(size_t j = 0; j < cols; ++j){
            p[i][j] = in.p[i][j];
        }
    }
    return *this;//*this;
}

const DenseMatrix DenseMatrix::operator+(const MyMatBase &in) const{
    assert_check::isEqual(rows, in.getRows(), __FILE__, __LINE__);
    DenseMatrix out(rows, cols);
    for(intptr_t i = out.rows - 1; i >= 0; --i){
        for(intptr_t j = out.cols - 1; j >= 0; --j){
            out.p[i][j] = p[i][j] + in.get(i, j);
        }
    }
    return out;
}

const DenseMatrix DenseMatrix::operator+(double_t in) const{
    if(in == 0){
        // return the object
        return *this;
    }
    DenseMatrix out(rows, cols);

    for(intptr_t i = out.rows - 1; i >= 0; --i){
        for(intptr_t j = out.cols - 1; j >= 0; --j){
            out.p[i][j] = p[i][j] + in;
        }
    }
    return out;
}

const DenseMatrix DenseMatrix::operator-(double_t in) const{
    DenseMatrix out(rows, cols);
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            out.p[i][j] = p[i][j] - in;
        }
    }
    return out;
}

const DenseMatrix DenseMatrix::operator-(const MyMatBase& in) const{
    DenseMatrix out(rows, cols);
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            out.p[i][j] = p[i][j] - in.get(i, j);
        }
    }
    return out;
}

const DenseMatrix DenseMatrix::operator*(const MyMatBase &in) const{
    assert_check::isEqual(cols, in.getRows(), __FILE__, __LINE__);
    DenseMatrix out(rows, in.getCols());
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            for(intptr_t k = cols - 1; k >= 0; --k){
                out.p[i][j] += p[i][k] * in.get(k, j);
            }
        }
    }
    return out;
}

//// this function is equal to the notation '.*' in matlab
//// you should use use this function like this:
////      DenseMatrix A(3,4,5);
////      DenseMatrix B(3,4,3);
////      cout << A*B.add(0);
//const DenseMatrix DenseMatrix::operator*(const DenseMatrix &&in) const{
//    DenseMatrix out(rows, cols);
//    for(intptr_t i = out.rows - 1; i >= 0; --i){
//        for(intptr_t j = out.cols - 1; j >= 0; --j){
//            out.p[i][j] = p[i][j] * in.p[i][j];
//        }
//    }
//    return out;
//}

const DenseMatrix DenseMatrix::operator*(double_t in) const{
    DenseMatrix out(rows, cols);
    for(intptr_t i = out.rows - 1; i >= 0; --i){
        for(intptr_t j = out.cols - 1; j >= 0; --j){
            out.p[i][j] = p[i][j] * in;
        }
    }
    return out;
}

const DenseMatrix DenseMatrix::operator/(const MyMatBase& in) const{
    DenseMatrix out(rows, cols);
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            out.p[i][j] = p[i][j] / in.get(i, j);
        }
    }
    return out;
}

const DenseMatrix DenseMatrix::operator/(double_t in) const{
    DenseMatrix out(rows, cols);
    for(intptr_t i = out.rows - 1; i >= 0; --i){
        for(intptr_t j = out.cols - 1; j >= 0; --j){
            out.p[i][j] = p[i][j] / in;
        }
    }
    return out;
}

const DenseMatrix DenseMatrix::operator()(size_t rs, size_t re, size_t cs, size_t ce){
    DenseMatrix tmp(re - rs, ce - cs);
    for(intptr_t i = re - rs + 1; i >= 0; --i){
        for(intptr_t j = ce - cs; j >= 0; --j){
            tmp.p[i][j] = p[i + rs][j + cs];
        }
    }
    return tmp;
}

const DenseMatrix DenseMatrix::T(void) const{
    DenseMatrix tmp(cols, rows);
    for(intptr_t i = tmp.rows - 1; i >= 0; --i){
        for(intptr_t j = tmp.cols - 1; j >= 0; --j){
            tmp.p[i][j] = p[j][i];
        }
    }
    return tmp;
}

inline void DenseMatrix::swap(double_t &A, double_t &B) const{
    double_t tmp;
    tmp = A;
    A = B;
    B = tmp;
}
//
bool DenseMatrix::set(size_t i, size_t j, double_t data){
    p[i][j] = data;
    return true;
}

bool DenseMatrix::setCol(size_t inum, double_t* pt, size_t num){
    for(size_t i = 0; i < num; ++i){
        p[num][inum] = pt[i];
    }
    return true;
}

bool DenseMatrix::setRow(size_t inum, double_t* pt, size_t num){
    for(size_t i = 0; i < num; ++i){
        p[inum][i] = pt[i];
    }
    return true;
}
//
double_t DenseMatrix::get(size_t i, size_t j) const{
    return p[i][j];
}
// I don't want to copy the data
// but I don't have the technology about the way
// just copy the pointer
// I must see some open source code such as numpy or OpenCV
// This function is different from Matlab, which use 0 as the
// smallest subscript, so remember that the end of the row or
// col should add 1 to get the real row or col.
DenseMatrix DenseMatrix::get(size_t rs, size_t re, size_t cs, size_t ce) const{
    DenseMatrix tmp(re - rs + 1, ce - cs + 1);
    for(intptr_t i = re - rs; i >= 0; --i){
        for(intptr_t j = ce - cs; j >= 0; --j){
            tmp.p[i][j] = p[i + rs][j + cs];
        }
    }
    return tmp;
}
// copy the data
bool DenseMatrix::copyTo(DenseMatrix &in) const{
    // release the memory
    if(this == &in){
        return true;
    } else{
        if(in.p != nullptr){
            in.clear();
        }
    }
    // copy the variable
    in.rows = rows;
    in.cols = cols;
    // then copy the data
    in.p = new double_t*[rows];
    for(size_t i = 0; i < rows; ++i){
        in.p[i] = new double_t[cols];
        for(size_t j = 0; j < cols; ++j){
            in.p[i][j] = p[i][j];
        }
    }
    return true;
}


ostream& operator<<(ostream& output, const DenseMatrix& _A){
    for(size_t i = 0; i < _A.getRows(); ++i){
        output << "     ";
        for(size_t j = 0; j < _A.getCols(); ++j){
            output << _A.get(i, j) << " , ";
        }
        output << "\n";
    }
    output << "\n";
    return output;
}

DenseMatrix DenseMatrix::ones(size_t _rows, size_t _cols){
    return DenseMatrix(_rows, _cols, 1);
}

DenseMatrix DenseMatrix::zeros(size_t _rows, size_t _cols){
    return DenseMatrix(_rows, _cols);
}

DenseMatrix DenseMatrix::eyes(size_t _rows){
    DenseMatrix tmp(_rows, _rows);
    for(size_t i = 0; i < _rows; ++i){
        tmp.set(i, i, 1);
    }
    return tmp;
}

bool DenseMatrix::Ri_Rj(size_t i, size_t j){
    bool error;
    error = assert_check::isLessthan(i, rows, __FILE__, __LINE__);
    error = assert_check::isLessthan(j, rows, __FILE__, __LINE__);
    if(!error){
        return false;
    }
    for(intptr_t num = cols - 1; num >= 0; --num){
        swap(p[i][num], p[j][num]);
    }

    return true;
}

bool DenseMatrix::Rix_k(size_t i, double_t j){
    bool error = assert_check::isLessthan(i, rows, __FILE__, __LINE__);
    if(!error){
        return false;
    }
    for(intptr_t num = cols - 1; num >= 0; --num){
        p[i][num] = p[i][num] * j;
    }
    return true;
}

bool DenseMatrix::Rki_j(size_t i, double_t k, size_t j){
    bool error = assert_check::isLessthan(i, rows, __FILE__, __LINE__);
    if(!error){
        return false;
    }
    for(intptr_t num = cols - 1; num >= 0; --num){
        p[j][num] += p[i][num] * k;
    }
    return true;
}

bool DenseMatrix::Ri_x(size_t i, double_t in){
    bool error = assert_check::isLessthan(i, rows, __FILE__, __LINE__);
    if(!error){
        return false;
    }
    for(intptr_t num = cols - 1; num >= 0; --num){
        p[i][num] *= in;
    }
    return true;
}

const DenseMatrix DenseMatrix::adj(void) const{
    assert_check::isEqual(rows, cols, __FILE__, __LINE__);
    DenseMatrix tmp(cols, rows);

    for(intptr_t i = cols - 1; i >= 0; --i){
        for(intptr_t j = rows - 1; j >= 0; --j){

        }
    }

    return tmp;
}

double_t DenseMatrix::det() const{
    double_t val = 1;
    double_t sig = 1;
    size_t pos = 0;

    DenseMatrix tmp(*this);

    for(size_t j = 0; j < cols; ++j){
        pos = find_num(j);
        if(pos == rows){
            return 0;
        }
        if(pos != j){
            tmp.Ri_Rj(j, pos);
            sig = 1 - sig;
        }
        if(tmp.p[j][j] != 1){
            val *= tmp.p[j][j];
            tmp.Ri_x(j, 1 / tmp.p[j][j]);
        }
        for(size_t k = 0; k < rows; ++k){
            if(k != j){
                if(tmp.p[k][j] != 0){
                    tmp.Rki_j(j, -tmp.p[k][j], k);
                }
            }
        }
    }
    return val*sig;
    // return....
}

const DenseMatrix DenseMatrix::inv(void) const{
    bool error = assert_check::isEqual(rows, cols, __FILE__, __LINE__);
    if(!error){
        return 0;
    }
    size_t pos = 0;
    DenseMatrix tmp(*this);
    DenseMatrix E = DenseMatrix::eyes(rows);
    double_t cache = 0;

    for(size_t j = 0; j < cols; ++j){
        pos = find_num(j);
        if(pos == rows){
            return 0;
        }
        if(pos != j){
            tmp.Ri_Rj(j, pos);
            E.Ri_Rj(j, pos);
        }
        if(tmp.p[j][j] != 1){
            cache = 1 / tmp.p[j][j];
            tmp.Ri_x(j, cache);
            E.Ri_x(j, cache);
        }
        for(size_t k = 0; k < rows; ++k){
            if(k != j){
                if(tmp.p[k][j] != 0){
                    cache = -tmp.p[k][j];
                    tmp.Rki_j(j, cache, k);
                    E.Rki_j(j, cache, k);
                }
            }
        }
    }
    return E;
}

size_t DenseMatrix::find_num(size_t i) const{
    size_t pos = i;
    for(; pos < rows; ++pos){
        if(p[pos][i] == 1){
            return pos;
        } else if(p[pos][i] != 0){
            return pos;
        }
    }
    return pos;
}

const DenseMatrix DenseMatrix::sum(bool axis = 0) const{
    if(rows == 1){
        double_t sum = 0;
        DenseMatrix out(1, 1);
        for(intptr_t i = cols - 1; i >= 0; --i){
            sum += p[0][i];
        }
        out.set(0, 0, sum);
        return out;
    } else if(cols == 1){
        double_t sum = 0;
        DenseMatrix out(1, 1);
        for(intptr_t j = rows - 1; j >= 0; --j){
            sum += p[j][0];
        }
        out.set(0, 0, sum);
        return out;
    } else{
        if(axis == 0){
            DenseMatrix out(1, cols);
            for(intptr_t i = cols - 1; i >= 0; --i){
                for(intptr_t j = rows - 1; j >= 0; --j){
                    out.set(0, i, out.get(0, i) + p[j][i]);
                }
            }
            return out;
        } else{
            DenseMatrix out(rows, 1);
            for(intptr_t i = rows - 1; i >= 0; --i){
                for(intptr_t j = cols - 1; j >= 0; --j){
                    out.set(i, 0, out.get(i, 0) + p[i][j]);
                }
            }
            return out;
        }
    }
}

const DenseMatrix DenseMatrix::log(void) const{
    DenseMatrix out(rows, cols);
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            out.set(i, j, std::log(p[i][j]));
        }
    }
    return out;
}

const DenseMatrix DenseMatrix::exp(void) const{
    DenseMatrix out(rows, cols);
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            out.set(i, j, std::exp(p[i][j]));
        }
    }
    return out;
}

const DenseMatrix DenseMatrix::pow(double_t in) const{
    DenseMatrix out(rows, cols);
    for(intptr_t i = rows - 1; i >= 0; --i){
        for(intptr_t j = cols - 1; j >= 0; --j){
            out.set(i, j, std::pow(p[i][j], in));
        }
    }
    return out;
}


}// env
