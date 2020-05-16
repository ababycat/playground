#include <iostream>

#include <string>
#include <stack>

using namespace std;

string ReverseSentence(string str) {
    string out;
    stack<string> word;

    for(int i=str.size()-1; i >= 0; --i){
        if(str[i] == ' '){
            while(!word.empty()){
                out += word.top();
                word.pop();
            }
            out += " ";
        }else{
            word.push(string(1, str[i]));
        }
    }
    while(!word.empty()){
        out += word.top();
        word.pop();
    }
    return out;
}


int main()
{
	cout << "--" << ReverseSentence("AB CD EF") << "--" << endl;
    return 0;
}
