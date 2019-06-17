#include <future>

namespace ppt
{
namespace shared
{

class scoped_thread
{
    std::thread t;

    public:
    explicit scoped_thread(std::thread t_) : t(std::move(t_))
    {
        if (!t.joinable())
            throw std::logic_error("thread is not joinable");
    }

    ~scoped_thread()
    {
        t.join();
    }

    scoped_thread(scoped_thread const&) = delete;
    scoped_thread& operator=(scoped_thread const&) = delete;
};

} // namespace shared
} // namespace ppt