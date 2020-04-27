# rm logs/*
for f in ./logs/blur*; do
    cp -- "$f" "./csvs/$(basename -- "$f" .log).csv"
done

