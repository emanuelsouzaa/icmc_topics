## Tuning com irace

Instale o pacote irace com o comando "install.packages("irace")"

Linux:

Rscript -e "library(irace); irace(scenario = readScenario('scenario.txt'))" | tee resultados_irace.txt

Windows

Rscript -e "library(irace); irace(scenario = readScenario('scenario.txt'))" | Tee-Object resultados_irace.txt