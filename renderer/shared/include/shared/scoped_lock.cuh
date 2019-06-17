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

    scoped_lock(const scoped_lock&) = delete;
    scoped_lock(scoped_lock&&) = delete;
    scoped_lock& operator=(const scoped_lock&) = delete;
    scoped_lock& operator=(scoped_lock&&) = delete;
};

} // namespace shared
} // namespace ppt
