#pragma once
#include <iomanip>
#include <iostream>
#include <string>
#include <time.h>
namespace ppt
{
namespace shared
{

/**
 * ScopedTimer can be used to time how long time a scope exists.
 * To use it, create a local ScopedTimer in your scope, and it will print at destruction time.
 */
struct scoped_timer
{
    const clock_t start;
    const std::string name;

    scoped_timer(const std::string& name) : start(clock()), name(name)
    {
    }

    ~scoped_timer()
    {
        std::clog << std::setw(25) << std::left << name << std::setw(12)
                  << ((double)(clock() - start)) / CLOCKS_PER_SEC << " seconds\n";
    }
};

} // namespace shared
} // namespace ppt