#include "job_sched_llm.h"

void main() {
	try {
		job_sched_llm llm;
		llm.load_prompts_from_file();
		llm.print_prompts();
		llm.generate_answer();
	} catch (const job_sched_exception& e) {
		std::cerr << e.what() << std::endl;
	}
}