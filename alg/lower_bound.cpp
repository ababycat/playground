#include <algorithm>
#include <vector>
#include <iostream>

using namespace std;

int main(){
    int a[6] = {1, 7, 3, 5, 9, 4};

    vector<int> list = vector<int>(a);
    int out = lower_bound(list.begin(), list.end(), 3);

    cout << out;
    return 0;
}

