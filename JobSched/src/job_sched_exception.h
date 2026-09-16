#include <string>
#include <stdexcept>

class job_sched_exception : public std::runtime_error
{
public:
	job_sched_exception(const std::string& msg) : std::runtime_error(msg) {}
};

