// #include "Catch2/catch.hpp"
#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file

#include <catch2/catch.hpp>
#include <shared/probability_helpers.cuh>
#include <vector>


TEST_CASE("Calculate mean for 1 elem is elem", "[calc_mean]")
{
    auto vals1 = std::vector<float>{ 5 };
    auto vals2 = std::vector<float>{ 123 };
    auto vals3 = std::vector<float>{ -34 };
    auto vals4 = std::vector<float>{ 0 };
    REQUIRE(ppt::shared::calc_mean(&vals1[0], 1) == 5.0f);
    REQUIRE(ppt::shared::calc_mean(&vals2[0], 1) == 123.0f);
    REQUIRE(ppt::shared::calc_mean(&vals3[0], 1) == -34.0f);
    REQUIRE(ppt::shared::calc_mean(&vals4[0], 1) == 0.0f);
}

TEST_CASE("Calculate mean for 0 elem is 0", "[calc_mean]")
{
    auto vals = std::vector<float>{};
    REQUIRE(ppt::shared::calc_mean(&vals[0], 0) == 0.0f);
}


TEST_CASE("Online Mean rollout", "[calc_mean]")
{
    auto vals = std::vector<float>{ 1, 1, 1, 2, 1, 1, 1, 1 };
    auto online_mean = ppt::shared::calc_mean_online_rollout(&vals[0], vals.size());
    auto offline_mean = ppt::shared::calc_mean(&vals[0], vals.size());

    REQUIRE(online_mean == Approx(offline_mean));
}

TEST_CASE("Print some online variances", "[calc_mean]")
{
    auto vals = std::vector<float>{ 1, 1, 1, 2, 1, 1, 1, 1 };
    auto online_vari = ppt::shared::calc_variance_online_rollout(&vals[0], vals.size());
    auto offline_vari = ppt::shared::calc_variance(&vals[0], vals.size());
    REQUIRE(online_vari == Approx(offline_vari));
}