#pragma once

#include <unordered_map>
#include <vector>
#include <iostream>


namespace learn_map{
    // TODO: Ã»×öÍê
    //// key: string
    //// value: int
    //#include <list>
    //#include <string>
    //#include <unordered_map>
    //#include <algorithm>
    //
    //class LRU{
    //
    //public:
    //	LRU(){}
    //	LRU(int capacity) : capacity(capacity){}
    //
    //	int capacity;
    //
    //	std::unordered_map<std::string, int> table;
    //
    //	std::list<std::string> link;
    //	// 
    //	int get(const std::string& key);
    //
    //	void set(const std::string& key, int value);
    //
    //}; 
    //
    //int LRU::get(const std::string& key)
    //{
    //
    //
    //	return table.at(key);
    //}
    //
    //void LRU::set(const std::string& key, int value){
    //	const std::unordered_map<std::string, int>::iterator iter = table.find(key);
    //	if(iter != table.end()){
    //		std::swap(iter, table.end());
    //		*iter = value;
    //		return;
    //	}
    //
    //	if(link.size() < capacity){
    //		table[key] = value;
    //		link.push_back(key);
    //		return;
    //	}
    //
    //	link.push_back(key);
    //	link.pop_front();
    //	table[key] = value;
    //
    //}



    using namespace std;

    vector<int> twoSum(vector<int>& nums, int target){
        vector<int> out;
        unordered_map<int, int> table;
        for(int idx = 0; idx < nums.size(); ++idx){
            auto iter = table.find(target - nums[idx]);
            if(iter != table.end()){
                return { idx, iter->second };
            }
            table.insert({ { nums[idx], idx } });
        }

        return {};

        //for(unsigned int idx = 0; idx < nums.size(); ++idx){
        //    table[nums[idx]] = idx;
        //}

        //for(unsigned int idx = 0; idx < nums.size(); ++idx){
        //    int sub = target - nums[idx];
        //    if(sub == nums[idx]){
        //        continue;
        //    }

        //    auto itor = table.find(sub);
        //    if(itor != table.end()){
        //        out.push_back(idx);
        //        out.push_back(itor->second);
        //        return out;
        //    }
        //}

        //return out;
    }

    int test_main(){

        vector<int> input = { 3, 3 };

        vector<int> output = twoSum(input, 6);

        for(auto out : output){
            cout << out << endl;
        }

        return 0;
    }

}
