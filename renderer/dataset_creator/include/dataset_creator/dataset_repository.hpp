#pragma once
#include <shared/render_datapoint.cuh>
#include <string>

namespace ppt
{
namespace dataset_creator
{

class dataset_repository
{

    public:
    const std::string datastore_path;
    const std::string run_name;

    dataset_repository(const std::string& datastore_path, const std::string& run_name)
      : datastore_path(datastore_path), run_name(run_name)
    {
    }

    void save_datapoint(const shared::render_datapoint& render_datapoint, const std::string& file_name);

    void save_dataset(const std::vector<shared::render_datapoint>& render_dataset);

    private:
    std::string get_file_name(const std::string& file_name, bool is_target, int render_number) const;
};

} // namespace dataset_creator
} // namespace ppt