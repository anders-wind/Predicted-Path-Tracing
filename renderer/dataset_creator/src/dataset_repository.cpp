#include <dataset_creator/dataset_repository.hpp>
#include <fstream>
#include <iomanip>
#include <shared/scoped_timer.cuh>
#include <sstream>

namespace ppt
{
namespace dataset_creator
{

std::string dataset_repository::get_file_name(const std::string& file_name,
                                              bool is_target,
                                              int render_number,
                                              std::string file_extension) const
{
    std::stringstream ss;
    ss << datastore_path << "/" << file_name << (is_target ? "_target" : "_input_");
    if (!is_target)
    {
        ss << std::setfill('0') << std::setw(2) << render_number;
    }
    ss << file_extension;
    return ss.str();
}

void dataset_repository::save_datapoint(const shared::render_datapoint& render_datapoint, const std::string& file_name)
{
    const auto timer = shared::scoped_timer("save_datapoint");
    std::ofstream target_file;
    target_file.open(get_file_name(file_name, true, 0, ".csv"));
    target_file << render_datapoint.get_target_string();
    target_file.close();

    for (size_t i = 0; i < render_datapoint.renders_size(); i++)
    {
        std::ofstream render_file;
        render_file.open(get_file_name(file_name, false, i, ".csv"));
        render_file << render_datapoint.get_render_string(i);
        render_file.close();
    }
}

void dataset_repository::save_ppm(const shared::render_datapoint& render_datapoint, const std::string& file_name)
{
    const auto timer = shared::scoped_timer("save_datapoint");
    std::ofstream target_file;
    target_file.open(get_file_name(file_name, true, 0, ".ppm"));
    target_file << render_datapoint.get_ppm_representation(render_datapoint.target);
    target_file.close();
}

void dataset_repository::save_datapoints(const std::vector<shared::render_datapoint>& render_dataset,
                                         const std::string& file_name)
{
    auto i = 0;
    std::stringstream ss;
    for (const auto& datapoint : render_dataset)
    {
        ss << file_name;
        ss << std::setfill('0') << std::setw(2) << i << "_";
        save_datapoint(datapoint, ss.str());
        ss.clear();
        i++;
    }
}

void dataset_repository::save_ppms(const std::vector<shared::render_datapoint>& render_dataset,
                                   const std::string& file_name)
{
    auto i = 0;
    std::stringstream ss;
    for (const auto& datapoint : render_dataset)
    {
        ss << file_name;
        ss << std::setfill('0') << std::setw(2) << i << "_";
        save_ppm(datapoint, ss.str());
        ss.clear();
        i++;
    }
}
} // namespace dataset_creator
} // namespace ppt