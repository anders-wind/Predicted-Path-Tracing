#pragma once
#include <memory>
#include <mutex>

namespace ppt
{
namespace shared
{

struct scoped_lock
{
    private:
    std::shared_ptr<std::mutex> lock;

    public:
    scoped_lock(const std::shared_ptr<std::mutex>& lock) : lock(lock)
    {
        while (!lock->try_lock())
        {
        }
    }
    ~scoped_lock()
    {
        lock->unlock();
    }
};

} // namespace shared
} // namespace ppt
