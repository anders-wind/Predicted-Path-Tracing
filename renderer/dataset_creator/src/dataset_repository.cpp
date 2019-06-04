#include <dataset_creator/dataset_repository.hpp>
#include <fstream>
#include <iomanip>
#include <shared/scoped_timer.cuh>
#include <sstream>

namespace ppt
{
namespace dataset_creator
{

// HELPERS

std::string dataset_repository::get_file_path(const std::string& file_name) const
{
    std::stringstream ss;
    ss << datastore_path << "/" << file_name;
    return ss.str();
}


std::string
dataset_repository::get_file_name(const std::string& name, bool is_target, int render_number, std::string file_extension) const
{
    std::stringstream ss;
    ss << name << (is_target ? "_target" : "_input_");
    if (!is_target)
    {
        ss << std::setfill('0') << std::setw(2) << render_number;
    }
    ss << file_extension;
    return ss.str();
}

void save_to_file(std::string file_path, std::string content)
{
    std::ofstream file;
    file.open(file_path);
    file << content;
    file.close(); // mixed oppinions on the web around if it is good practise to do this or not (since ofstream is RAII)
}

// PUBLIC METHODS

void dataset_repository::save_datapoint(const shared::render_datapoint& render_datapoint, const std::string& file_name)
{
    const auto timer = shared::scoped_timer("save_datapoint");
    std::stringstream datapoint_ss;

    auto target_file_name = get_file_name(file_name, true, 0, ".csv");
    save_to_file(get_file_path(target_file_name), render_datapoint.get_target_string());

    datapoint_ss << target_file_name;
    for (auto i = 0; i < render_datapoint.renders_size(); i++)
    {
        auto render_file_name = get_file_name(file_name, false, i, ".csv");
        save_to_file(get_file_path(render_file_name), render_datapoint.get_render_string(i));
        datapoint_ss << ", " << render_file_name;
    }

    save_to_file(get_file_path(file_name + ".dp"), datapoint_ss.str());
}

void dataset_repository::save_datapoints(const std::vector<shared::render_datapoint>& render_dataset,
                                         const std::string& file_name)
{
    auto i = 0;
    std::stringstream ss;
    for (const auto& datapoint : render_dataset)
    {
        ss << file_name;
        ss << std::setfill('0') << std::setw(2) << i;
        save_datapoint(datapoint, ss.str());
        ss.str("");
        ss.clear();
        i++;
    }
}

void dataset_repository::save_ppm(const shared::render_datapoint& render_datapoint, const std::string& file_name)
{
    const auto timer = shared::scoped_timer("save_ppm");

    auto target_file_name = get_file_name(file_name, true, 0, ".ppm");
    save_to_file(get_file_path(target_file_name),
                 render_datapoint.get_ppm_representation(render_datapoint.target));

    for (size_t i = 0; i < render_datapoint.renders_size(); i++)
    {
        std::ofstream render_file;
        auto render_file_name = get_file_name(file_name, false, i, ".ppm");
        save_to_file(get_file_path(render_file_name),
                     render_datapoint.get_ppm_representation(render_datapoint.renders[i]));
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
        ss.str("");
        ss.clear();
        i++;
    }
}
} // namespace dataset_creator
} // namespace ppt