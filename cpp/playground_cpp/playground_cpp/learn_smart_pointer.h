#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <list>
#include <algorithm>


namespace learn_smart_pointer{

    using namespace std;

    int global_index = 0;

    class Test{
    public:
        int a;
        Test() :a(0){
            ++global_index;
            cout << "dfconstruct:" << a << endl;
        }

        Test(int a) :a(a){
            ++global_index;
            cout << "nmconstruct:" << a << endl;
        }

        Test(const Test& test){
            this->a = test.a;
            ++global_index;
            cout << "cpconstruct:" << a << endl;
        }

        Test(Test&& test) : a(test.a){
            cout << "mvconstruct:" << a << endl;
        }

        Test& operator=(const Test& test){
            this->a = test.a;
            ++global_index;
            cout << "=operator:" << a << endl;
            return *this;
        }

        friend ostream& operator<<(ostream& _cout, Test& a){
            _cout << a.a;
            return _cout;
        }

        void print(){
            cout << "this a: " << this->a << endl;
        }

        ~Test(){
            cout << "destruct:" << this->a << endl;
        }
    };


    shared_ptr<Test> test_func(){
        // 使用智能指针，自动调用析构函数
        shared_ptr<Test> sp_test = make_shared<Test>();

        // 手动调用析构函数
        //Test* test = new Test();
        //delete test;
        //test = 0;
        return sp_test;
    }

    // 使用默认指针
    //Test foo_bar(Test test){
    //    Test local = test, * heap = new Test(test);
    //    delete heap;
    //    *heap = local;
    //    Test t[2] = { local, *heap };
    //    return *heap;
    //}

    // 使用智能指针
    Test foo_bar(Test test){
        Test local = test;
        shared_ptr<Test> heap = make_shared<Test>(test);
        *heap = local;
        Test t[2] = { local, *heap };
        return *heap;
    }

    void test_const_ref(const Test& a)
    {
        cout << a.a << endl;
    }

    void move_test(Test& a, Test& b){
        Test tmp(std::move(a));
        a = std::move(b);
        b = std::move(tmp);
    }



    int main_test()
    {
        //vector<Test> t;
        //t.push_back(Test(1));
        //t.push_back(Test(2));

        //Test a = Test(3);
        //cout << a << endl;

        std::unordered_map<std::string, int> table;

        table["a"] = 1;
        table["b"] = 10;
        std::cout << table["a"] << ", " << table["b"] << std::endl;

        auto iter1 = table.find("a");
        auto iter2 = table.find("b");
        std::swap(iter1, iter2);

        //std::swap(table["a"], table["b"]);

        std::cout << table["a"] << ", " << table["b"] << std::endl;

        //vector<string> v1;
        //{
        //    vector<string> v2 = {"a", "b", "c"};
        //    v1 = v2;
        //}

        //for(auto v : v1){
        //    cout << v << endl;
        //}

        //Test a1(9);
        //Test a2(10);
        //cout << "r a1: " << &a1 << endl;
        //cout << "r a2: " << &a2 << endl;

        //move_test(a1, a2);
        //cout << "r a1: " << &a1 << endl;
        //cout << "r a2: " << &a2 << endl;

        //cout << "a1: " << a1.a << endl;
        //cout << "a2: " << a2.a << endl;

        //Test a(10);
        //test_const_ref(a);


        //cout << "test func begin" << endl;
        //shared_ptr<Test> out = test_func();
        //cout << "test func end" << endl;

        //Test a(1);
        //shared_ptr<Test> p = make_shared<Test>(1);

        //auto f = [p](){
        //    cout << p->a << endl;
        //};

        //f();

        //Test a(1);
        //auto f = [a](){
        //    cout << a.a << endl;
        //};

        //f();
        //Test a;
        //Test b = foo_bar(a);


        //cout << b.a << endl;
        //// 向量测试
        //std::vector<Test> p;
        //
        //Test t;

        //std::cout << p.capacity() << ',' << sizeof(p) << endl;

        //p.resize(1);

        //std::cout << p.capacity() << ',' << sizeof(p) << endl;

        //std::cout << "test" << endl;

        //std::list<Test> p;

        //Test t;
        //p.push_back(t);
        //p.resize(10);


        return 0;
    }



}
