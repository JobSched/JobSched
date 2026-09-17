#include <iostream>
#include <vector>
#include <string>
#include "llama.h"

#include <fstream>
#include <sstream>

#include "job_sched_exception.h"


class job_sched_llm
{
	llama_model* model;
	llama_model_params model_parameter;

	llama_context* context;
	llama_context_params context_parameter;

	const llama_vocab* vocab;

	std::string system_prompt;
	std::string user_prompt;
	std::string prompt;

	std::vector<llama_token> tokens;

public:
	job_sched_llm() {
		init();
	}

	// set parameters for model and context
	void init() {
		llama_backend_init();

		// disable default llama logging
		llama_log_set([](auto, const char* text, auto) {}, nullptr);

		model_parameter = llama_model_default_params();
		std::string model_path = "models/Phi-3-mini-128k-instruct_f16.gguf";

		context_parameter = llama_context_default_params();
		context_parameter.n_ctx = 1024 * 12;

		std::cout << "Loading LLM... ";

		// load LLM
		model = llama_model_load_from_file(model_path.c_str(), model_parameter);
		if (!model) {
			throw job_sched_exception("Could not load the LLM.");
		}

		// create context
		context = llama_init_from_model(model, context_parameter);
		if (!context) {
			llama_model_free(model);
			throw job_sched_exception("Could not create LLM context.");
		}

		// get Vocabulary from the LLM
		vocab = llama_model_get_vocab(model);

		std::cout << "-> completed" << std::endl;
	}

	void load_prompts_from_file() {
		std::cout << "Loading prompts... ";

		// load system prompt from file
		std::ifstream system_prompt_file("prompts/system_prompt.txt");
		std::stringstream system_prompt_buffer;
		system_prompt_buffer << system_prompt_file.rdbuf();
		system_prompt = system_prompt_buffer.str();

		// load user prompt from file
		std::ifstream user_prompt_file("prompts/user_prompt.txt");
		std::stringstream user_prompt_buffer;
		user_prompt_buffer << user_prompt_file.rdbuf();
		user_prompt = user_prompt_buffer.str();

		std::cout << "-> completed" << std::endl;
	}

	void print_prompts() {
		std::cout << "System Prompt: " << std::endl;
		std::cout << system_prompt << std::endl << std::endl;
		std::cout << "User Prompt: " << std::endl;
		std::cout << user_prompt << std::endl << std::endl;
	}

	void apply_chat_template() {
		// create vector for the chat messages
		std::vector<llama_chat_message> messages;

		// add system promt to the messages
		std::vector<char> system_prompt_vector(system_prompt.begin(), system_prompt.end());
		system_prompt_vector.push_back('\0');
		messages.push_back({ "system", system_prompt_vector.data() });

		// add user prompt to the messages
		std::vector<char> user_prompt_vector(user_prompt.begin(), user_prompt.end());
		user_prompt_vector.push_back('\0');
		messages.push_back({ "user",  user_prompt_vector.data() });

		// apply acutal llama chat template
		std::vector<char> messages_buffer(messages.size() * 2);
		int32_t messages_buffer_length = llama_chat_apply_template(nullptr, messages.data(), messages.size(), true, messages_buffer.data(), messages_buffer.size());

		if (messages_buffer_length > (int)messages_buffer.size()) {
			messages_buffer.resize(messages_buffer_length);
			messages_buffer_length = llama_chat_apply_template(nullptr, messages.data(), messages.size(), true, messages_buffer.data(), messages_buffer.size());
		}
		prompt = std::string(messages_buffer.begin(), messages_buffer.begin() + messages_buffer_length);
	}

	void tokenize_prompt() {
		std::vector<llama_token> prompt_tokens(prompt.length() + 100);
		int prompt_size = llama_tokenize(vocab, prompt.c_str(), prompt.length(), prompt_tokens.data(), prompt_tokens.size(), false, false);
		prompt_tokens.resize(prompt_size);
		tokens.insert(tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
	}

	void generate_answer() {
		apply_chat_template();
		tokenize_prompt();

		// create and adjust sampler chain
		llama_sampler* sampler;
		sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

		llama_sampler_chain_add(sampler, llama_sampler_init_top_k(50));
		llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
		//llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
		llama_sampler_chain_add(sampler, llama_sampler_init_temp(1.0f));
		llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));

		if (!sampler) {
			llama_free(context);
			llama_model_free(model);
			throw job_sched_exception("Could not create sampler.");
		}

		std::cout << "Answer: " << std::endl;

		// decode tokens of the prompt
		llama_decode(context, llama_batch_get_one(tokens.data(), tokens.size()));
		
		// loop for generating the answer tokens
		for (int i = 0; i < 500; i++) {
			// generate new token
			llama_token new_token = llama_sampler_sample(sampler, context, -1);
			llama_sampler_accept(sampler, new_token);

			// terminate, when the newly generated token is an end token
			if (llama_vocab_is_eog(vocab, new_token)) {
				break;
			}

			// add the newly generated token to the tokens vector and print it
			tokens.push_back(new_token);
			char new_token_buffer[128];
			int new_token_length = llama_token_to_piece(vocab, new_token, new_token_buffer, sizeof(new_token_buffer), 0, false);
			if (new_token_length > 0) {
				std::cout << std::string(new_token_buffer, new_token_length);
			}

			// decode the newly generated token
			llama_decode(context, llama_batch_get_one(&new_token, 1));
		}
		std::cout << std::endl;

		// free resources
		llama_sampler_free(sampler);
		llama_free(context);
		llama_model_free(model);
		llama_backend_free();
	}
};

