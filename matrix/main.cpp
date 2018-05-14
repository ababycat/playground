#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <set>



#include "mystring.h"
#include "mymat.h"
#include "testmat.h"

using namespace env;

using namespace std;




int main(){
    DenseMatrix A(3, 3, 5);
    DenseMatrix B;

    cout << A.getCols() << endl;
    cout << A.getRows() << endl;
    cout << A;
    cout << A.isSqure() << "\n";
    cout << sizeof A << "\n";

    // you should not use A again
    B = A;
    DenseMatrix C(3, 3, 5);

    //// cout << A.isMatInit() << "\n";
    //// cout << A;
    //// cout << sizeof(unsigned char);
    //
    // cout << B.getCols() <<  endl;
    // cout << B.getRows() <<  endl;
    DenseMatrix D = B.add(C);
    cout << D;

    cout << D.get(2, 2) << "\n";
    cout << D.get(0, 2, 0, 1).getRows();

    double_t p[4] = {1, 5, 6, 8};
    DenseMatrix Kp(p, size_t(4), 1);
    cout << Kp;

    double_t p2[5][5] = {{17, 24, 1, 8, 15}, {23, 5, 7, 14, 16}, {4, 6, 13, 20, 22}, {10, 12, 19, 21, 3}, {11, 18, 25, 2, 9}};
    DenseMatrix kp2(5, 5);
    kp2.setRow(0, p2[0], 5);
    kp2.setRow(1, p2[1], 5);
    kp2.setRow(2, p2[2], 5);
    kp2.setRow(3, p2[3], 5);
    kp2.setRow(4, p2[4], 5);
    cout << kp2;
    cout << kp2.det() / 100000.0 << endl;

    DenseMatrix E(5, 5);
    E.set(2, 0, 199);
    E.set(2, 1, 27);
    cout << E;
    E.Ri_Rj(2, 4);
    cout << E;


    DenseMatrix b(3, 3, 2);
    cout << b;
    b = 0;
    cout << b;

    DenseMatrix d(2, 2, 1);
    d.set(0, 1, 0);
    cout << d.det() << endl;
    cout << d;
    cout << d.inv();

    int data;
    cin >> data;
    while(data){
        cin >> data;
    }

    return 0;
}
