#include <iostream>
using namespace std;

int find(int *a, int len, int n)
{
    int left(0),right(len),mid = (left+right)/2;
    while(left <= right)
    {
        if(n > a[mid]) left = mid + 1;
        else if(n < a[mid]) right = mid - 1;
        else return mid;
        mid = (left + right)/2;
    }
    return left;
}

const int n = 9;

int main()
{
    int i, j, len;
    int c[100]={0};
    // cin >> n;
    
    // for(int i = 0;i < n;i++)
    //     cin >> a[i];
    int a[n] = {98,94,71,74,30,2,38,91,9};

    c[0] = -1;
    c[1] = a[0];
    len = 1;
        
    for(i = 1; i <= n; i++)
    {
        j = find(c, len, a[i]);
        c[j] = a[i];
        if(j > len)
            len = j;
        
        cout << "J" << j;
        cout << "[";
        for(int t=0; t < n; ++t){
            cout << c[t] << ", ";
        }
        cout << "]" << endl;
    }
    
    cout << "max len: " << len << endl;
    return 0;
}