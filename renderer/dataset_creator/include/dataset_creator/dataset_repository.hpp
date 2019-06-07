#pragma once
#include <experimental/filesystem>
#include <path_tracer/render_datapoint.cuh>
#include <string>

namespace ppt
{
namespace dataset_creator
{

class dataset_repository
{

    public:
    const std::string datastore_path;

    dataset_repository(const std::string& datastore_path) : datastore_path(datastore_path)
    {
        std::experimental::filesystem::create_directories({ datastore_path });
    }

    void save_datapoint(const path_tracer::render_datapoint& render_datapoint, const std::string& file_name);
    void save_datapoints(const std::vector<path_tracer::render_datapoint>& render_dataset,
                         const std::string& file_name);

    void save_ppm(const path_tracer::render_datapoint& render_datapoint, const std::string& file_name);
    void save_ppms(const std::vector<path_tracer::render_datapoint>& render_dataset, const std::string& file_name);


    private:
    std::string get_file_path(const std::string& file_name) const;
    std::string get_file_name(const std::string& name, bool is_target, int render_number, std::string file_extension) const;
};

} // namespace dataset_creator
} // namespace ppt