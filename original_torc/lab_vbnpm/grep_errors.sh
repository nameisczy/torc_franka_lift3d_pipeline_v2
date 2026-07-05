# grep -lR '[A-Za-z ]Error' experiments/runs | cut -d\/ -f1-5 > DELETE_ME
grep --include="output.csv" -lR '[A-Za-z ]Error' experiments/runs | tee DELETE_ME
