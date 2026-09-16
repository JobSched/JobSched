# Job Scheduling mit LLM

## Ordnerstruktur

- Die selbsterstellten Quellcodedateien befinden sich im Order *JobSched/src*.
- Die Modell gguf-Dateien können im Order *JobSched/models* abgelegt werden.
- Die txt-Dateien für die Prompts können im Order *JobSched/prompts* abgelegt werden.
- Die h-Dateien und lib-Dateien von llama.cpp befinden sich im Ordner *JobSched/3rd/llama.cpp*.
- Das Programm wurde unter Windows compiliert. Die notwendigen ddl-Dateien befinden sich in *x64/Debug* und als Backup noch in *JobSched/3rd/llama.cpp/bin*.

## Auf des Programmcodes (*JobSched/src*)

- Das Programm wird von *main.h* aus gestartet.
- In *main.h* wird eine Instanz der Klasse *job_sched_llm* initialisiert.
- Die Klasse *job_sched_llm* übernimmt dann das eigentliche Laden des LLMs, die Vorbereitungs der Prompts und die eigentliche Anfrage ans LLM.