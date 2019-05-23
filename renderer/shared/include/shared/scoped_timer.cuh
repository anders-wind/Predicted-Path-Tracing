#include <iostream>
#include <string>
#include <time.h>

namespace ppt
{
namespace shared
{

struct ScopedTimer
{
    const clock_t start;
    const std::string name;
    ScopedTimer(const std::string& name) : start(clock()), name(name)
    {
    }

    ~ScopedTimer()
    {
        std::clog << name << " took " << ((double)(clock() - start)) / CLOCKS_PER_SEC << " seconds.\n";
    }
};

} // namespace shared
} // namespace ppt