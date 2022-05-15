#include "pch.h"
#include "CppUnitTest.h"

#include "../../playground_cpp/playground_cpp/learn_play.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;





namespace playgroundunittest
{
	TEST_CLASS(playgroundunittest)
	{
	public:
		
		TEST_METHOD(TestName)
		{
			Logger::WriteMessage("In TestName");
			using namespace learn_play;
			std::string name = "hello";
			auto n = Name(name);
			Assert::AreEqual(n.GetName(), name);
		}

		TEST_METHOD(TestCopyNotSame)
		{
			Logger::WriteMessage("TestCopyNotSame");
			using namespace learn_play;
			std::string name = "hello";
			auto n = Name(name);
			Assert::AreNotSame(n.GetName(), name);
		}
	};
}
