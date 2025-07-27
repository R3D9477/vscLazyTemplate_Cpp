#define BOOST_TEST_MODULE test_project_version

#include "project_version.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(test_project_version)

BOOST_AUTO_TEST_CASE(test_valid_project_version)
{
    BOOST_CHECK(project_version() != "");
}
}
