#include "pch.h"
#include "CppUnitTest.h"
#include <algorithm>


using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace playgroundunittest{

	class TestMoveConstruct{
	public:
		static int default_num;
		static int move_num;
		static int copy_num;
		TestMoveConstruct(){
			Logger::WriteMessage("Default Construct\n");
			++default_num;
		}


		TestMoveConstruct(const TestMoveConstruct& copy){
			Logger::WriteMessage("Copy Construct\n");
			++copy_num;
		}

		TestMoveConstruct(TestMoveConstruct&& Temp){
			Logger::WriteMessage("Move Construct\n");
			++move_num;
		}
	};

	int TestMoveConstruct::default_num = 0;
	int TestMoveConstruct::move_num = 0;
	int TestMoveConstruct::copy_num = 0;

	class TestA{
	public:
		TestMoveConstruct test;
		TestA(TestMoveConstruct&& t): test(std::move(t)){
			Logger::WriteMessage("test A Construct\n");
		}
	};

	class TestB{
	public:
		TestMoveConstruct test;
		TestB(TestMoveConstruct&& t) : test(t){
			Logger::WriteMessage("test B Construct\n");
		}
	};

	class TestDefaultInitializeClass
	{
	public:
		TestDefaultInitializeClass()
		{

		}

		int i;
		int a : 1;
		int b : 1;
		int c : 2;
	};

	enum ETestEnum
	{
		First,
		Second
	};

	void test_no_name_enum(ETestEnum)
	{
		Logger::WriteMessage("test_no_name_enum function run!\n");
	}

	void test_no_name_int()
	{
		Logger::WriteMessage("test_no_name_int() function run!\n");
	}

	void test_no_name_int(int)
	{
		Logger::WriteMessage("test_no_name_int(int) function run!\n");
	}

	TEST_CLASS(test_cpp){
public:

	TEST_METHOD(TestStaticCast)
	{
		int i = 10;
		float a = 10.4f;
		float b = 10.5f;

		float c = 10.9f;

		std::string s;

		Assert::AreEqual(i, static_cast<int>(a));

		Assert::AreEqual(a, static_cast<float>(i), static_cast<float>(1));

		Assert::AreEqual(i, static_cast<int>(b));

		Assert::AreNotEqual(i + 1, static_cast<int>(c));

	}

	TEST_METHOD(TestMove)
	{

		TestMoveConstruct t;
		
		Assert::AreEqual(TestMoveConstruct::default_num, 1);
		Assert::AreEqual(TestMoveConstruct::move_num, 0);

		TestA ta(std::move(t));
		Assert::AreEqual(TestMoveConstruct::default_num, 1);
		Assert::AreEqual(TestMoveConstruct::move_num, 1);

		TestB tb(std::move(t));
		Assert::AreEqual(TestMoveConstruct::default_num, 1);
		Assert::AreEqual(TestMoveConstruct::move_num, 1);
		Assert::AreEqual(TestMoveConstruct::copy_num, 1);
	}

    TEST_METHOD(TestDefaultInitialize)
    {

		int i(0);

        Assert::AreEqual(i, 0);

		int b;
		Assert::AreNotEqual(b, 0);

		Logger::WriteMessage("b not initialized\n");
		
		TestDefaultInitializeClass c;
		Assert::AreNotEqual(c.i, 0);

		Logger::WriteMessage("c initialized\n");
    }

	TEST_METHOD(TestNoNameEnum)
	{

		test_no_name_enum(ETestEnum::First);
		test_no_name_int(1);
	}
	};
}
