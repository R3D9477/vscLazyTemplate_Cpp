#pragma once

#include <sstream>

class CppInfo
{
public:
    std::string GetLanguageVersion()
    {
        switch(__cplusplus)
        {
            case 1:
                return "pre-C++98";
                break;
            case 199711L:
                return "C++98";
                break;
            case 201103L:
                return "C++11";
                break;
            case 201402L:
                return "C++14";
                break;
            case 201703L:
                return "C++17";
                break;
            case 202002L:
                return "C++20";
                break;
            case 202302L:
                return "C++23";
                break;
            default:
                return "Unknown";
                break;
        }
    }
    std::string GetCompilerVersion()
    {
        std::stringstream compilerInfoStream;

#if defined(__clang__)
        compilerInfoStream << "Clang++ v" << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__) || defined(__GNUG__)
        compilerInfoStream << "GCC v" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#elif defined(_MSC_VER)
        compilerInfoStream << "MSVC++ v";
        if (_MSC_VER >= 1900)
        {
            compilerInfoStream << (_MSC_VER/100-5) << "." << (_MSC_VER-1900);
        }
        else if (_MSC_VER >= 1100)
        {
            compilerInfoStream << (_MSC_VER/100-4) << "." << 0;
        }
        else
        {
            compilerInfoStream << 0 << "." << 0;
        }
#else
        compilerInfoStream << "Unknown";
#endif
        return compilerInfoStream.str();
    }
};