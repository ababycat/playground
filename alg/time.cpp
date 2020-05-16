#include <iostream>
#include <ctime>
#include <math.h>

using namespace std;

int frequency_of_primes(int n)
{
    int i, j;
    int freq = n - 1;
    for (i = 2; i <= n; ++i)
        for (j = sqrt(i); j > 1; --j)
            if (i % j == 0)
            {
                --freq;
                break;
            }
    return freq;
}

int main()
{
    clock_t t;
    int f;
    t = clock();
    //   printf ("Calculating...\n");
    //   f = frequency_of_primes (99999);
    //   printf ("The number of primes lower than 100,000 is: %d\n",f);
    for(int i=0; i< 10000000; ++i)
        f = 10 * 1024;
    t = clock() - t;
    printf("The number of primes lower than 100,000 is: %d\n", f);
    printf("It took me %d clicks (%.9f seconds).\n", t, ((float)t) / CLOCKS_PER_SEC);
    return 0;
}